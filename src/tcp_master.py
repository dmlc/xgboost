"""
Master script for xgboost, tcp_master
This script can be used to start jobs of multi-node xgboost using sync_tcp

Tianqi Chen
"""

import sys
import os
import socket
import struct
import subprocess
from threading import Thread

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

# magic number used to verify existence of data
kMagic = 0xff99

class Master:
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
        return ['master_uri=%s' % socket.gethostname(),
                'master_port=%s' % self.port]    
    def accept_slaves(self, nslave):        
        slave_addrs = []
        for rank in range(nslave):
            while True:
                fd, s_addr = self.sock.accept()
                slave = ExSocket(fd)
                nparent = int(rank != 0)
                nchild = 0
                if (rank + 1) * 2 - 1 < nslave:
                    nchild += 1
                if (rank + 1) * 2 < nslave:
                    nchild += 1                
                try:
                    magic = slave.recvint()
                    if magic != kMagic:
                        print 'invalid magic number=%d from %s' % (magic, s_addr[0])                        
                        slave.sock.close()
                        continue
                except socket.error:
                    print 'sock error in %s' % (s_addr[0])
                    slave.sock.close()
                    continue
                slave.sendint(kMagic)
                slave.sendint(rank)
                slave.sendint(nslave)
                slave.sendint(nparent)
                slave.sendint(nchild)
                if nparent != 0:
                    parent_index = (rank + 1) / 2 - 1
                    ptuple = slave_addrs[parent_index]
                    slave.sendstr(ptuple[0])
                    slave.sendint(ptuple[1])
                s_port = slave.recvint()
                assert rank == len(slave_addrs)
                slave_addrs.append((s_addr[0], s_port))
                slave.sock.close()
                print 'finish starting rank=%d at %s' % (rank, s_addr[0])
                break
        print 'all slaves setup complete'
        
def mpi_submit(nslave, args):
    cmd = ' '.join(['mpirun -n %d' % nslave] + args)
    print cmd
    return subprocess.check_call(cmd, shell = True)
    
def submit(nslave, args, fun_submit = mpi_submit):
    master = Master()
    submit_thread = Thread(target = fun_submit, args = (nslave, args + master.slave_args()))
    submit_thread.start()
    master.accept_slaves(nslave)
    submit_thread.join()
