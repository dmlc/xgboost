"""
Tracker script for DMLC
Implements the tracker control protocol
 - start dmlc jobs
 - start ps scheduler and rabit tracker
 - help nodes to establish links with each other

Tianqi Chen
"""
# pylint: disable=invalid-name, missing-docstring, too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
from __future__ import absolute_import

import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread

class ExSocket(object):
    """
    Extension of socket to handle recv and send of special data
    """
    def __init__(self, sock):
        self.sock = sock
    def recvall(self, nbytes):
        res = []
        nread = 0
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024))
            nread += len(chunk)
            res.append(chunk)
        return b''.join(res)
    def recvint(self):
        return struct.unpack('@i', self.recvall(4))[0]
    def sendint(self, n):
        self.sock.sendall(struct.pack('@i', n))
    def sendstr(self, s):
        self.sendint(len(s))
        self.sock.sendall(s.encode())
    def recvstr(self):
        slen = self.recvint()
        return self.recvall(slen).decode()

# magic number used to verify existence of data
kMagic = 0xff99

def get_some_ip(host):
    return socket.getaddrinfo(host, None)[0][4][0]

def get_family(addr):
    return socket.getaddrinfo(addr, None)[0][0]

class SlaveEntry(object):
    def __init__(self, sock, s_addr):
        slave = ExSocket(sock)
        self.sock = slave
        self.host = get_some_ip(s_addr[0])
        magic = slave.recvint()
        assert magic == kMagic, 'invalid magic number=%d from %s' % (magic, self.host)
        slave.sendint(kMagic)
        self.rank = slave.recvint()
        self.world_size = slave.recvint()
        self.jobid = slave.recvstr()
        self.cmd = slave.recvstr()
        self.wait_accept = 0
        self.port = None

    def decide_rank(self, job_map):
        if self.rank >= 0:
            return self.rank
        if self.jobid != 'NULL' and self.jobid in job_map:
            return job_map[self.jobid]
        return -1

    def assign_rank(self, rank, wait_conn, tree_map, parent_map, ring_map):
        self.rank = rank
        nnset = set(tree_map[rank])
        rprev, rnext = ring_map[rank]
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
        if rprev != -1 and rprev != rank:
            nnset.add(rprev)
            self.sock.sendint(rprev)
        else:
            self.sock.sendint(-1)
        # send next link
        if rnext != -1 and rnext != rank:
            nnset.add(rnext)
            self.sock.sendint(rnext)
        else:
            self.sock.sendint(-1)
        while True:
            ngood = self.sock.recvint()
            goodset = set([])
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
                self.sock.sendint(wait_conn[r].port)
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

class RabitTracker(object):
    """
    tracker for rabit
    """
    def __init__(self, hostIP, nslave, port=9091, port_end=9999):
        sock = socket.socket(get_family(hostIP), socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind((hostIP, port))
                self.port = port
                break
            except socket.error as e:
                if e.errno in [98, 48]:
                    continue
                else:
                    raise
        sock.listen(256)
        self.sock = sock
        self.hostIP = hostIP
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.nslave = nslave
        logging.info('start listen on %s:%d', hostIP, self.port)

    def __del__(self):
        self.sock.close()

    @staticmethod
    def get_neighbor(rank, nslave):
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank // 2 - 1)
        if rank * 2 - 1 < nslave:
            ret.append(rank * 2 - 1)
        if rank * 2 < nslave:
            ret.append(rank * 2)
        return ret

    def slave_envs(self):
        """
        get enviroment variables for slaves
        can be passed in as args or envs
        """
        return {'DMLC_TRACKER_URI': self.hostIP,
                'DMLC_TRACKER_PORT': self.port}

    def get_tree(self, nslave):
        tree_map = {}
        parent_map = {}
        for r in range(nslave):
            tree_map[r] = self.get_neighbor(r, nslave)
            parent_map[r] = (r + 1) // 2 - 1
        return tree_map, parent_map

    def find_share_ring(self, tree_map, parent_map, r):
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
        nset = set(tree_map[r])
        cset = nset - set([parent_map[r]])
        if len(cset) == 0:
            return [r]
        rlst = [r]
        cnt = 0
        for v in cset:
            vlst = self.find_share_ring(tree_map, parent_map, v)
            cnt += 1
            if cnt == len(cset):
                vlst.reverse()
            rlst += vlst
        return rlst

    def get_ring(self, tree_map, parent_map):
        """
        get a ring connection used to recover local data
        """
        assert parent_map[0] == -1
        rlst = self.find_share_ring(tree_map, parent_map, 0)
        assert len(rlst) == len(tree_map)
        ring_map = {}
        nslave = len(tree_map)
        for r in range(nslave):
            rprev = (r + nslave - 1) % nslave
            rnext = (r + 1) % nslave
            ring_map[rlst[r]] = (rlst[rprev], rlst[rnext])
        return ring_map

    def get_link_map(self, nslave):
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        tree_map, parent_map = self.get_tree(nslave)
        ring_map = self.get_ring(tree_map, parent_map)
        rmap = {0 : 0}
        k = 0
        for i in range(nslave - 1):
            k = ring_map[k][1]
            rmap[k] = i + 1

        ring_map_ = {}
        tree_map_ = {}
        parent_map_ = {}
        for k, v in ring_map.items():
            ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
        for k, v in tree_map.items():
            tree_map_[rmap[k]] = [rmap[x] for x in v]
        for k, v in parent_map.items():
            if k != 0:
                parent_map_[rmap[k]] = rmap[v]
            else:
                parent_map_[rmap[k]] = -1
        return tree_map_, parent_map_, ring_map_

    def accept_slaves(self, nslave):
        # set of nodes that finishs the job
        shutdown = {}
        # set of nodes that is waiting for connections
        wait_conn = {}
        # maps job id to rank
        job_map = {}
        # list of workers that is pending to be assigned rank
        pending = []
        # lazy initialize tree_map
        tree_map = None

        while len(shutdown) != nslave:
            fd, s_addr = self.sock.accept()
            s = SlaveEntry(fd, s_addr)
            if s.cmd == 'print':
                msg = s.sock.recvstr()
                logging.info(msg.strip())
                continue
            if s.cmd == 'shutdown':
                assert s.rank >= 0 and s.rank not in shutdown
                assert s.rank not in wait_conn
                shutdown[s.rank] = s
                logging.debug('Recieve %s signal from %d', s.cmd, s.rank)
                continue
            assert s.cmd == 'start' or s.cmd == 'recover'
            # lazily initialize the slaves
            if tree_map is None:
                assert s.cmd == 'start'
                if s.world_size > 0:
                    nslave = s.world_size
                tree_map, parent_map, ring_map = self.get_link_map(nslave)
                # set of nodes that is pending for getting up
                todo_nodes = list(range(nslave))
            else:
                assert s.world_size == -1 or s.world_size == nslave
            if s.cmd == 'recover':
                assert s.rank >= 0

            rank = s.decide_rank(job_map)
            # batch assignment of ranks
            if rank == -1:
                assert len(todo_nodes) != 0
                pending.append(s)
                if len(pending) == len(todo_nodes):
                    pending.sort(key=lambda x: x.host)
                    for s in pending:
                        rank = todo_nodes.pop(0)
                        if s.jobid != 'NULL':
                            job_map[s.jobid] = rank
                        s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                        if s.wait_accept > 0:
                            wait_conn[rank] = s
                        logging.debug('Recieve %s signal from %s; assign rank %d',
                                      s.cmd, s.host, s.rank)
                if len(todo_nodes) == 0:
                    logging.info('@tracker All of %d nodes getting started', nslave)
                    self.start_time = time.time()
            else:
                s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                logging.debug('Recieve %s signal from %d', s.cmd, s.rank)
                if s.wait_accept > 0:
                    wait_conn[rank] = s
        logging.info('@tracker All nodes finishes job')
        self.end_time = time.time()
        logging.info('@tracker %s secs between node start and job finish',
                     str(self.end_time - self.start_time))

    def start(self, nslave):
        def run():
            self.accept_slaves(nslave)
        self.thread = Thread(target=run, args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        while self.thread.isAlive():
            self.thread.join(100)

class PSTracker(object):
    """
    Tracker module for PS
    """
    def __init__(self, hostIP, cmd, port=9091, port_end=9999, envs=None):
        """
        Starts the PS scheduler
        """
        self.cmd = cmd
        if cmd is None:
            return
        envs = {} if envs is None else envs
        self.hostIP = hostIP
        sock = socket.socket(get_family(hostIP), socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind(('', port))
                self.port = port
                sock.close()
                break
            except socket.error:
                continue
        env = os.environ.copy()

        env['DMLC_ROLE'] = 'scheduler'
        env['DMLC_PS_ROOT_URI'] = str(self.hostIP)
        env['DMLC_PS_ROOT_PORT'] = str(self.port)
        for k, v in envs.items():
            env[k] = str(v)
        self.thread = Thread(
            target=(lambda: subprocess.check_call(self.cmd, env=env, shell=True)), args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        if self.cmd is not None:
            while self.thread.isAlive():
                self.thread.join(100)

    def slave_envs(self):
        if self.cmd is None:
            return {}
        else:
            return {'DMLC_PS_ROOT_URI': self.hostIP,
                    'DMLC_PS_ROOT_PORT': self.port}


def get_host_ip(hostIP=None):
    if hostIP is None or hostIP == 'auto':
        hostIP = 'ip'

    if hostIP == 'dns':
        hostIP = socket.getfqdn()
    elif hostIP == 'ip':
        from socket import gaierror
        try:
            hostIP = socket.gethostbyname(socket.getfqdn())
        except gaierror:
            logging.warn('gethostbyname(socket.getfqdn()) failed... trying on hostname()')
            hostIP = socket.gethostbyname(socket.gethostname())
        if hostIP.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            hostIP = s.getsockname()[0]
    return hostIP


def submit(nworker, nserver, fun_submit, hostIP='auto', pscmd=None):
    if nserver == 0:
        pscmd = None

    envs = {'DMLC_NUM_WORKER' : nworker,
            'DMLC_NUM_SERVER' : nserver}
    hostIP = get_host_ip(hostIP)

    if nserver == 0:
        rabit = RabitTracker(hostIP=hostIP, nslave=nworker)
        envs.update(rabit.slave_envs())
        rabit.start(nworker)
    else:
        pserver = PSTracker(hostIP=hostIP, cmd=pscmd, envs=envs)
        envs.update(pserver.slave_envs())
    fun_submit(nworker, nserver, envs)

    if nserver == 0:
        rabit.join()
    else:
        pserver.join()

def start_rabit_tracker(args):
    """Standalone function to start rabit tracker.

    Parameters
    ----------
    args: arguments to start the rabit tracker.
    """
    envs = {'DMLC_NUM_WORKER' : args.num_workers,
            'DMLC_NUM_SERVER' : args.num_servers}
    rabit = RabitTracker(hostIP=get_host_ip(args.host_ip), nslave=args.num_workers)
    envs.update(rabit.slave_envs())
    rabit.start(args.num_workers)
    sys.stdout.write('DMLC_TRACKER_ENV_START\n')
    # simply write configuration to stdout
    for k, v in envs.items():
        sys.stdout.write('%s=%s\n' % (k, str(v)))
    sys.stdout.write('DMLC_TRACKER_ENV_END\n')
    sys.stdout.flush()
    rabit.join()


def main():
    """Main function if tracker is executed in standalone mode."""
    parser = argparse.ArgumentParser(description='Rabit Tracker start.')
    parser.add_argument('--num-workers', required=True, type=int,
                        help='Number of worker proccess to be launched.')
    parser.add_argument('--num-servers', default=0, type=int,
                        help='Number of server process to be launched. Only used in PS jobs.')
    parser.add_argument('--host-ip', default=None, type=str,
                        help=('Host IP addressed, this is only needed ' +
                              'if the host IP cannot be automatically guessed.'))
    parser.add_argument('--log-level', default='INFO', type=str,
                        choices=['INFO', 'DEBUG'],
                        help='Logging level of the logger.')
    args = parser.parse_args()

    fmt = '%(asctime)s %(levelname)s %(message)s'
    if args.log_level == 'INFO':
        level = logging.INFO
    elif args.log_level == 'DEBUG':
        level = logging.DEBUG
    else:
        raise RuntimeError("Unknown logging level %s" % args.log_level)

    logging.basicConfig(format=fmt, level=level)

    if args.num_servers == 0:
        start_rabit_tracker(args)
    else:
        raise RuntimeError("Do not yet support start ps tracker in standalone mode.")

if __name__ == "__main__":
    main()
