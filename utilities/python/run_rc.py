import subprocess

cmd_list=["../../build/release/run_rc",
          "rcNetwork.json", # path to JSON file
          "../../data/pythonExample.txt", # path to data file
          "0", # gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
          "0", # percentage of additive noise
          "spikeLog.bin", # name of output spike file
          "potentialLog.bin", # name of output potential file
          "1", # Enable GUI (bool)
          "10", #timestep
          "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"  # either leave this field blank -> potential logs all the reservoir, or, any number that goes after the timestep argument will be tracked. so you can add "0", "1", "2" to track the neurons with ID 0, 1 and 2
        ]

subprocess.call(cmd_list)

# NOTES: If a network has too many events, it might be faster to run the network clock-based
