# create Ti64 material with penred

import pyPenred

pyPenred.simulation.materials.create(name = "Ti64",
                                     density = 4.43,
                                     composition = [(22,0.90), #90% Ti
                                                    (13, 0.06), # 6% Aluminum
                                                     (23, 0.04)], # 4% Vanadium
                                     filename =r"E:\Code\EPBF_Twin\testing\meltpool\ti64.mat") 