import subprocess

cmd_list=["../../build/release/generate_rc",
          "28", # pixel grid width (int) 
          "28", # pixel grid height (int)
          "0", # gaussian mean for input weights (float)
          "1", # gaussian standard deviation for input weights (float)
          "1000", # number of neurons inside the reservoir (int) 
          "0", # gaussian mean for weights (float)
          "1", # gaussian standard deviation for weights (float)
          "1", # percentage likelihood of feedforward connections (int)
          "1", # percentage likelihood of feedback connections (int) 
          "1", # percentage likelihood of self-excitation (int)
          "100", # current step function reset value (int)
          "120", # potential decay (int)
          "0", # refractory period (int)
          "0", # winner-takes-all (0 or 1 for true or false)
          "0", # threshold adaptation to firing rate
          "1", # 0 for event-based, > 0 for clock-based
        ]

subprocess.call(cmd_list)


##### COMMENTS FROM THE OLD FILE (PARAMETERS ARE NOT THE SAME - CHECK FOR THE CHANGES) #####

# subprocess.call(["../../build/release/rc", "../../data/pythonExample.txt", "spikeLog.bin", "potentialLog.bin", "28", "28", "10", "1.0", "0", "100", "0", "0", "10", "20", "3", "0"])
# subprocess.call(["../../build/release/rc", "../../data/pythonExample.txt", "spikeLog.bin", "potentialLog.bin", "28", "28", "10", "1.0", "0", "100", "0", "0", "10", "20", "3", "0", "0", "1", "0", "10"])
# subprocess.call(["../../build/release/rc", "../../data/pythonExample.txt", "spikeLog.bin", "potentialLog.bin", "28", "28", "10", "1.0", "0", "100", "0", "0", "10", "20", "3", "0", "0", "1", "0", "10", "1"])
