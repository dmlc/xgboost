"""
Tracker script for rabit
Implements the tracker control protocol 
 - start rabit jobs
 - help nodes to establish links with each other

Tianqi Chen
"""

import sys
import os
import socket
import struct
import subprocess
import random
from threading import Thread

"""
Extension of socket to handle recv and send of special data
"""
class ExSocket:    
    def __init__(self, sock):
        self.sock = sock
    def recvall(self, nbytes):
        res = []
        sock = self.sock
        nread = 0    
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024), socket.MSG_WAITALL)
            nread += len(chunk)
            res.append(chunk)
        return ''.join(res)
    def recvint(self):
        return struct.unpack('@i', self.recvall(4))[0]
    def sendint(self, n):
        self.sock.sendall(struct.pack('@i', n))
    def sendstr(self, s):
        self.sendint(len(s))
        self.sock.sendall(s)
    def recvstr(self):
        slen = self.recvint()
        return self.recvall(slen)

# magic number used to verify existence of data
kMagic = 0xff99

class SlaveEntry:
    def __init__(self, sock, s_addr):
        slave = ExSocket(sock)
        self.sock = slave
        self.host = s_addr[0]
        magic = slave.recvint()
        assert magic == kMagic, 'invalid magic number=%d from %s' % (magic, s_addr[0])
        slave.sendint(kMagic)
        self.rank = slave.recvint()
        self.jobid = slave.recvstr()
        self.cmd = slave.recvstr()

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
            for i in xrange(ngood):
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
    
class Tracker:
    def __init__(self, port = 9000, port_end = 9999):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind(('', port))
                self.port = port
                break
            except socket.error:
                continue
        sock.listen(16)
        self.sock = sock        
        print 'start listen on %s:%d' % (socket.gethostname(), self.port)
    def __del__(self):
        self.sock.close()
    def slave_args(self):
        return ['rabit_tracker_uri=%s' % socket.gethostname(),
                'rabit_tracker_port=%s' % self.port]        
    def get_neighbor(self, rank, nslave):
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank / 2 - 1)
        if rank * 2 - 1  < nslave:
            ret.append(rank * 2 - 1)            
        if rank * 2 < nslave:
            ret.append(rank * 2)
        return ret
    def get_tree(self, nslave):
        tree_map = {}
        parent_map = {}
        for r in range(nslave):
            tree_map[r] = self.get_neighbor(r, nslave)
            parent_map[r] = (r + 1) / 2 - 1
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
    def accept_slaves(self, nslave):
        tree_map, parent_map = self.get_tree(nslave)
        ring_map = self.get_ring(tree_map, parent_map)
        # set of nodes that finishs the job
        shutdown = {}
        # set of nodes that is waiting for connections
        wait_conn = {}
        # set of nodes that is pending for getting up
        todo_nodes = range(nslave)
        random.shuffle(todo_nodes)
        # maps job id to rank
        job_map = {}
        # list of workers that is pending to be assigned rank
        pending = []
        
        while len(shutdown) != nslave:
            fd, s_addr = self.sock.accept()
            s = SlaveEntry(fd, s_addr)
            if s.cmd == 'shutdown':
                assert s.rank >= 0 and s.rank not in shutdown
                assert s.rank not in wait_conn
                shutdown[s.rank] = s
                continue
            assert s.cmd == 'start' or s.cmd == 'recover'
            if s.cmd == 'recover':
                assert s.rank >= 0
                print 'Recieve recover signal from %d' % s.rank
            rank = s.decide_rank(job_map)
            if rank == -1:
                assert len(todo_nodes) != 0
                rank = todo_nodes.pop(0)
                if s.jobid != 'NULL':
                    job_map[s.jobid] = rank                
            s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
            if s.wait_accept > 0:
                wait_conn[rank] = s            
        print 'All nodes finishes job'

def mpi_submit(nslave, args):
    cmd = ' '.join(['mpirun -n %d' % nslave] + args)
    print cmd
    return subprocess.check_call(cmd, shell = True)
    
def submit(nslave, args, fun_submit = mpi_submit):
    master = Tracker()
    submit_thread = Thread(target = fun_submit, args = (nslave, args + master.slave_args()))
    submit_thread.start()
    master.accept_slaves(nslave)
    submit_thread.join()
