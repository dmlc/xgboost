# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-branches
"""
This script is a variant of dmlc-core/dmlc_tracker/tracker.py,
which is a specialized version for xgboost tasks.
"""
import argparse
import logging
import socket
import struct
import sys
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Union

_RingMap = Dict[int, Tuple[int, int]]
_TreeMap = Dict[int, List[int]]


class ExSocket:
    """
    Extension of socket to handle recv and send of special data
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def recvall(self, nbytes: int) -> bytes:
        """Receive number of bytes."""
        res = []
        nread = 0
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024))
            nread += len(chunk)
            res.append(chunk)
        return b"".join(res)

    def recvint(self) -> int:
        """Receive an integer of 32 bytes"""
        return struct.unpack("@i", self.recvall(4))[0]

    def sendint(self, value: int) -> None:
        """Send an integer of 32 bytes"""
        self.sock.sendall(struct.pack("@i", value))

    def sendstr(self, value: str) -> None:
        """Send a Python string"""
        self.sendint(len(value))
        self.sock.sendall(value.encode())

    def recvstr(self) -> str:
        """Receive a Python string"""
        slen = self.recvint()
        return self.recvall(slen).decode()


# magic number used to verify existence of data
MAGIC_NUM = 0xFF99


def get_some_ip(host: str) -> str:
    """Get ip from host"""
    return socket.getaddrinfo(host, None)[0][4][0]


def get_family(addr: str) -> int:
    """Get network family from address."""
    return socket.getaddrinfo(addr, None)[0][0]


class WorkerEntry:
    """Hanlder to each worker."""

    def __init__(self, sock: socket.socket, s_addr: Tuple[str, int]):
        worker = ExSocket(sock)
        self.sock = worker
        self.host = get_some_ip(s_addr[0])
        magic = worker.recvint()
        assert magic == MAGIC_NUM, f"invalid magic number={magic} from {self.host}"
        worker.sendint(MAGIC_NUM)
        self.rank = worker.recvint()
        self.world_size = worker.recvint()
        self.task_id = worker.recvstr()
        self.cmd = worker.recvstr()
        self.wait_accept = 0
        self.port: Optional[int] = None

    def print(self, use_logger: bool) -> None:
        """Execute the print command from worker."""
        msg = self.sock.recvstr()
        # On dask we use print to avoid setting global verbosity.
        if use_logger:
            logging.info(msg.strip())
        else:
            print(msg.strip(), flush=True)

    def decide_rank(self, job_map: Dict[str, int]) -> int:
        """Get the rank of current entry."""
        if self.rank >= 0:
            return self.rank
        if self.task_id != "NULL" and self.task_id in job_map:
            return job_map[self.task_id]
        return -1

    def assign_rank(
        self,
        rank: int,
        wait_conn: Dict[int, "WorkerEntry"],
        tree_map: _TreeMap,
        parent_map: Dict[int, int],
        ring_map: _RingMap,
    ) -> List[int]:
        """Assign the rank for current entry."""
        self.rank = rank
        nnset = set(tree_map[rank])
        rprev, next_rank = ring_map[rank]
        self.sock.sendint(rank)
        # send parent rank
        self.sock.sendint(parent_map[rank])
        # send world size
        self.sock.sendint(len(tree_map))
        self.sock.sendint(len(nnset))
        # send the rprev and next link
        for r in nnset:
            self.sock.sendint(r)
        # send prev link
        if rprev not in (-1, rank):
            nnset.add(rprev)
            self.sock.sendint(rprev)
        else:
            self.sock.sendint(-1)
        # send next link
        if next_rank not in (-1, rank):
            nnset.add(next_rank)
            self.sock.sendint(next_rank)
        else:
            self.sock.sendint(-1)

        return self._get_remote(wait_conn, nnset)

    def _get_remote(
        self, wait_conn: Dict[int, "WorkerEntry"], nnset: Set[int]
    ) -> List[int]:
        while True:
            ngood = self.sock.recvint()
            goodset = set()
            for _ in range(ngood):
                goodset.add(self.sock.recvint())
            assert goodset.issubset(nnset)
            badset = nnset - goodset
            conset = []
            for r in badset:
                if r in wait_conn:
                    conset.append(r)
            self.sock.sendint(len(conset))
            self.sock.sendint(len(badset) - len(conset))
            for r in conset:
                self.sock.sendstr(wait_conn[r].host)
                port = wait_conn[r].port
                assert port is not None
                # send port of this node to other workers so that they can call connect
                self.sock.sendint(port)
                self.sock.sendint(r)
            nerr = self.sock.recvint()
            if nerr != 0:
                continue
            self.port = self.sock.recvint()
            rmset = []
            # all connection was successuly setup
            for r in conset:
                wait_conn[r].wait_accept -= 1
                if wait_conn[r].wait_accept == 0:
                    rmset.append(r)
            for r in rmset:
                wait_conn.pop(r, None)
            self.wait_accept = len(badset) - len(conset)
            return rmset


class RabitTracker:
    """
    tracker for rabit
    """

    def __init__(
        self,
        host_ip: str,
        n_workers: int,
        port: int = 0,
        use_logger: bool = False,
        sortby: str = "host",
    ) -> None:
        """A Python implementation of RABIT tracker.

        Parameters
        ..........
        use_logger:
            Use logging.info for tracker print command.  When set to False, Python print
            function is used instead.

        sortby:
            How to sort the workers for rank assignment. The default is host, but users
            can set the `DMLC_TASK_ID` via RABIT initialization arguments and obtain
            deterministic rank assignment. Available options are:
              - host
              - task

        """
        sock = socket.socket(get_family(host_ip), socket.SOCK_STREAM)
        sock.bind((host_ip, port))
        self.port = sock.getsockname()[1]
        sock.listen(256)
        self.sock = sock
        self.host_ip = host_ip
        self.thread: Optional[Thread] = None
        self.n_workers = n_workers
        self._use_logger = use_logger
        self._sortby = sortby
        logging.info("start listen on %s:%d", host_ip, self.port)

    def __del__(self) -> None:
        if hasattr(self, "sock"):
            self.sock.close()

    @staticmethod
    def _get_neighbor(rank: int, n_workers: int) -> List[int]:
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank // 2 - 1)
        if rank * 2 - 1 < n_workers:
            ret.append(rank * 2 - 1)
        if rank * 2 < n_workers:
            ret.append(rank * 2)
        return ret

    def worker_envs(self) -> Dict[str, Union[str, int]]:
        """
        get environment variables for workers
        can be passed in as args or envs
        """
        return {"DMLC_TRACKER_URI": self.host_ip, "DMLC_TRACKER_PORT": self.port}

    def _get_tree(self, n_workers: int) -> Tuple[_TreeMap, Dict[int, int]]:
        tree_map: _TreeMap = {}
        parent_map: Dict[int, int] = {}
        for r in range(n_workers):
            tree_map[r] = self._get_neighbor(r, n_workers)
            parent_map[r] = (r + 1) // 2 - 1
        return tree_map, parent_map

    def find_share_ring(
        self, tree_map: _TreeMap, parent_map: Dict[int, int], rank: int
    ) -> List[int]:
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from rank
        """
        nset = set(tree_map[rank])
        cset = nset - {parent_map[rank]}
        if not cset:
            return [rank]
        rlst = [rank]
        cnt = 0
        for v in cset:
            vlst = self.find_share_ring(tree_map, parent_map, v)
            cnt += 1
            if cnt == len(cset):
                vlst.reverse()
            rlst += vlst
        return rlst

    def get_ring(self, tree_map: _TreeMap, parent_map: Dict[int, int]) -> _RingMap:
        """
        get a ring connection used to recover local data
        """
        assert parent_map[0] == -1
        rlst = self.find_share_ring(tree_map, parent_map, 0)
        assert len(rlst) == len(tree_map)
        ring_map: _RingMap = {}
        n_workers = len(tree_map)
        for r in range(n_workers):
            rprev = (r + n_workers - 1) % n_workers
            rnext = (r + 1) % n_workers
            ring_map[rlst[r]] = (rlst[rprev], rlst[rnext])
        return ring_map

    def get_link_map(self, n_workers: int) -> Tuple[_TreeMap, Dict[int, int], _RingMap]:
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        tree_map, parent_map = self._get_tree(n_workers)
        ring_map = self.get_ring(tree_map, parent_map)
        rmap = {0: 0}
        k = 0
        for i in range(n_workers - 1):
            k = ring_map[k][1]
            rmap[k] = i + 1

        ring_map_: _RingMap = {}
        tree_map_: _TreeMap = {}
        parent_map_: Dict[int, int] = {}
        for k, v in ring_map.items():
            ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
        for k, tree_nodes in tree_map.items():
            tree_map_[rmap[k]] = [rmap[x] for x in tree_nodes]
        for k, parent in parent_map.items():
            if k != 0:
                parent_map_[rmap[k]] = rmap[parent]
            else:
                parent_map_[rmap[k]] = -1
        return tree_map_, parent_map_, ring_map_

    def _sort_pending(self, pending: List[WorkerEntry]) -> List[WorkerEntry]:
        if self._sortby == "host":
            pending.sort(key=lambda s: s.host)
        elif self._sortby == "task":
            pending.sort(key=lambda s: s.task_id)
        return pending

    def accept_workers(self, n_workers: int) -> None:
        """Wait for all workers to connect to the tracker."""

        # set of nodes that finishes the job
        shutdown: Dict[int, WorkerEntry] = {}
        # set of nodes that is waiting for connections
        wait_conn: Dict[int, WorkerEntry] = {}
        # maps job id to rank
        job_map: Dict[str, int] = {}
        # list of workers that is pending to be assigned rank
        pending: List[WorkerEntry] = []
        # lazy initialize tree_map
        tree_map = None

        while len(shutdown) != n_workers:
            fd, s_addr = self.sock.accept()
            s = WorkerEntry(fd, s_addr)
            if s.cmd == "print":
                s.print(self._use_logger)
                continue
            if s.cmd == "shutdown":
                assert s.rank >= 0 and s.rank not in shutdown
                assert s.rank not in wait_conn
                shutdown[s.rank] = s
                logging.debug("Received %s signal from %d", s.cmd, s.rank)
                continue
            assert s.cmd in ("start", "recover")
            # lazily initialize the workers
            if tree_map is None:
                assert s.cmd == "start"
                if s.world_size > 0:
                    n_workers = s.world_size
                tree_map, parent_map, ring_map = self.get_link_map(n_workers)
                # set of nodes that is pending for getting up
                todo_nodes = list(range(n_workers))
            else:
                assert s.world_size in (-1, n_workers)
            if s.cmd == "recover":
                assert s.rank >= 0

            rank = s.decide_rank(job_map)
            # batch assignment of ranks
            if rank == -1:
                assert todo_nodes
                pending.append(s)
                if len(pending) == len(todo_nodes):
                    pending = self._sort_pending(pending)
                    for s in pending:
                        rank = todo_nodes.pop(0)
                        if s.task_id != "NULL":
                            job_map[s.task_id] = rank
                        s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                        if s.wait_accept > 0:
                            wait_conn[rank] = s
                        logging.debug(
                            "Received %s signal from %s; assign rank %d",
                            s.cmd,
                            s.host,
                            s.rank,
                        )
                if not todo_nodes:
                    logging.info("@tracker All of %d nodes getting started", n_workers)
            else:
                s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                logging.debug("Received %s signal from %d", s.cmd, s.rank)
                if s.wait_accept > 0:
                    wait_conn[rank] = s
        logging.info("@tracker All nodes finishes job")

    def start(self, n_workers: int) -> None:
        """Strat the tracker, it will wait for `n_workers` to connect."""

        def run() -> None:
            self.accept_workers(n_workers)

        self.thread = Thread(target=run, args=(), daemon=True)
        self.thread.start()

    def join(self) -> None:
        """Wait for the tracker to finish."""
        while self.thread is not None and self.thread.is_alive():
            self.thread.join(100)

    def alive(self) -> bool:
        """Wether the tracker thread is alive"""
        return self.thread is not None and self.thread.is_alive()


def get_host_ip(host_ip: Optional[str] = None) -> str:
    """Get the IP address of current host.  If `host_ip` is not none then it will be
    returned as it's

    """
    if host_ip is None or host_ip == "auto":
        host_ip = "ip"

    if host_ip == "dns":
        host_ip = socket.getfqdn()
    elif host_ip == "ip":
        from socket import gaierror

        try:
            host_ip = socket.gethostbyname(socket.getfqdn())
        except gaierror:
            logging.debug(
                "gethostbyname(socket.getfqdn()) failed... trying on hostname()"
            )
            host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't have to be reachable
            s.connect(("10.255.255.255", 1))
            host_ip = s.getsockname()[0]

    assert host_ip is not None
    return host_ip


def start_rabit_tracker(args: argparse.Namespace) -> None:
    """Standalone function to start rabit tracker.

    Parameters
    ----------
    args: arguments to start the rabit tracker.
    """
    envs = {"DMLC_NUM_WORKER": args.num_workers, "DMLC_NUM_SERVER": args.num_servers}
    rabit = RabitTracker(
        host_ip=get_host_ip(args.host_ip), n_workers=args.num_workers, use_logger=True
    )
    envs.update(rabit.worker_envs())
    rabit.start(args.num_workers)
    sys.stdout.write("DMLC_TRACKER_ENV_START\n")
    # simply write configuration to stdout
    for k, v in envs.items():
        sys.stdout.write(f"{k}={v}\n")
    sys.stdout.write("DMLC_TRACKER_ENV_END\n")
    sys.stdout.flush()
    rabit.join()


def main() -> None:
    """Main function if tracker is executed in standalone mode."""
    parser = argparse.ArgumentParser(description="Rabit Tracker start.")
    parser.add_argument(
        "--num-workers",
        required=True,
        type=int,
        help="Number of worker process to be launched.",
    )
    parser.add_argument(
        "--num-servers",
        default=0,
        type=int,
        help="Number of server process to be launched. Only used in PS jobs.",
    )
    parser.add_argument(
        "--host-ip",
        default=None,
        type=str,
        help=(
            "Host IP addressed, this is only needed "
            + "if the host IP cannot be automatically guessed."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG"],
        help="Logging level of the logger.",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(message)s"
    if args.log_level == "INFO":
        level = logging.INFO
    elif args.log_level == "DEBUG":
        level = logging.DEBUG
    else:
        raise RuntimeError(f"Unknown logging level {args.log_level}")

    logging.basicConfig(format=fmt, level=level)

    if args.num_servers == 0:
        start_rabit_tracker(args)
    else:
        raise RuntimeError("Do not yet support start ps tracker in standalone mode.")


if __name__ == "__main__":
    main()
