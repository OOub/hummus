import subprocess

cmd_list=["../../build/release/run_rc",
          "../../data/testSave.json", # path to JSON file
          "../../data/pythonExample.txt", # path to data file
          "0", # gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
          "0", # percentage of additive noise
          "spikeLog.bin", # name of output spike file
          "potentialLog.bin", # name of output potential file
          "1", # Enable GUI (bool)
          "0", #timestep
               # either leave this field blank -> potential logs all the reservoir, or, any number that goes after the timestep argument will be tracked. so you can add "0", "1", "2" to track the neurons with ID 0, 1 and 2
        ]

subprocess.call(cmd_list)
 
