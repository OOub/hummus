import subprocess
import numpy as np

N = 1000
W = (np.random.randn(N,N)*0.2*np.random.binomial(n=1,p=0.01,size=(N,N)))
W = np.triu(W)
# np.sa
px = 28
U = 0.1*np.random.randn(px**2,N)*np.random.binomial(n=1,p=0.01,size=((px**2,N)))
np.savetxt("U.txt",U)
np.savetxt("W.txt",W)
cmd_list=["../../build/release/generate_rc",
          "28", # pixel grid width (int) 
          "28", # pixel grid height (int)
          "0", # gaussian mean for input weights (float)
          "0.2", # gaussian standard deviation for input weights (float)
          str(N), # number of neurons inside the reservoir (int) 
          "0", # gaussian mean for weights (float)
          "0.1", # gaussian standard deviation for weights (float)
          "2", # percentage likelihood of feedforward connections (int)
          "1", # percentage likelihood of feedback connections (int)
          "1", # percentage likelihood of self-excitation (int)
          "100", # current step function reset value (int)
          "120", # potential decay (int)
          "0", # refractory period (int)
          "1", # winner-takes-all (0 or 1 for true or false)
          "0", # threshold adaptation to firing rate
          "1", # bool 1 to use weight matrix, 0 to use the probabilities thing
          "./U.txt", # input weight matrix filename
          "./W.txt", # reservoir weight matrix filename
        ]

subprocess.call(cmd_list)
netname= "rcNetwork_N({})_gp({},{})_g({},{})_c({},{},{})_IPR({},{},{})_w{}.json".format(cmd_list[5],cmd_list[3],cmd_list[4],cmd_list[6],cmd_list[7],cmd_list[8],cmd_list[9],cmd_list[10],cmd_list[11],cmd_list[12],cmd_list[13],cmd_list[14])
print (netname)
subprocess.call(["mv","rcNetwork.json",netname])
