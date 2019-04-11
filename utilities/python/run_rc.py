import subprocess, os
datapath=os.path.expanduser("~/Datasets/N-MNIST/TrainTXT/3/00087.txt") 
cmd_list=["../../build/release/run_rc",
          "rcNetwork_N(100)_gp(0,0.2)_g(0,0.1)_c(2,1,1)_IPR(100,120,0)_w1.json", # path to JSON file
          datapath,#"../../data/pythonExample.txt", # path to data file
          "0", # gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
          "1", # percentage of additive noise
          "spikeLog.bin", # name of output spike file
          "potentialLog.bin", # name of output potential file
          "1", # Enable GUI (bool)
          "0", #timestep
          "1", #verbose
               # either leave this field blank -> potential logs all the reservoir, or, any number that goes after the timestep argument will be tracked. so you can add "0", "1", "2" to track the neurons with ID 0, 1 and 2
        ]

print(len(cmd_list))
subprocess.call(cmd_list)

# NOTES: If a network has too many events, it might be faster to run the network clock-based
