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
          "1", # winner-takes-all (0 or 1 for true or false)
          "0", # threshold adaptation to firing rate
        ]

subprocess.call(cmd_list)