import os
import argparse
import sys

def main():
  parser = argparse.ArgumentParser(description='TODO')
  parser.add_argument('-ho', '--host_dir', required=True)
  parser.add_argument('-s', '--submit_script', required=True)
  parser.add_argument('-rex', '--rabit_exec', required=True)
  parser.add_argument('-mpi', '--mpi_exec', required=True)
  args = parser.parse_args()

  ndata = [10**4, 10**5, 10**6, 10**7]
  nrepeat = [10**4, 10**3, 10**2, 10]

  machines = [2,4,8,16,31]

  executables = [args.rabit_exec, args.mpi_exec]

  for executable in executables:
    sys.stderr.write('Executable %s' % executable)
    sys.stderr.flush()
    for i, data in enumerate(ndata):
      for machine in machines:
        host_file = os.path.join(args.host_dir, 'hosts%d' % machine)
        cmd = 'python %s %d %s %s %d %d' % (args.submit_script, machine, host_file, executable, data, nrepeat[i])
        sys.stderr.write('data=%d, repeat=%d, machine=%d\n' % (data, nrepeat[i], machine))
        sys.stderr.flush()
        os.system(cmd)
    sys.stderr.write('\n')
    sys.stderr.flush()

if __name__ == "__main__":
  main()
