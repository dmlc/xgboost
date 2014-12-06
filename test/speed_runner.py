import os
import argparse

def main():
  parser = argparse.ArgumentParser(description='TODO')
  parser.add_argument('-ho', '--host_dir', required=True)
  parser.add_argument('-s', '--submit_script', required=True)
  args = parser.parse_args()

  ndata = [10^4, 10^5, 10^6, 10^7, 10^8]
  nrepeat = [10^2, 10^3, 10^4, 10^5]

  machines = [2,4,8,16,31]

  for data in ndata:
    for repeat in nrepeat:
      for machine in machines:
        host_file = os.path.join(args.host_dir, 'host%d' % machine)
        cmd = 'python %s %d %s %d %d' % (args.submit_script, machine, host_file, data, repeat)
        print 'data=%d, repeat=%d, machine=%d' % (data, repeat, machine)
        os.system(cmd)

if __name__ == "__main__":
  main()