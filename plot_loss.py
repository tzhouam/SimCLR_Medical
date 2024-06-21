import re
import matplotlib.pyplot as plt

# Your original string containing loss values
# ResNet18 Supervised
original_str = """
Epoch 1/100, Loss: 2.2463605403900146, Top1 Train Accuracy: 3.5714287757873535, Top5 Train Accuracy: 45.53571701049805             
Top1 Test Accuracy: 5.388017654418945, Top5 Test Accuracy: 87.41455078125                                                          
                                                                                                                                   
Epoch 2/100, Loss: 1.4398335218429565, Top1 Train Accuracy: 55.357147216796875, Top5 Train Accuracy: 91.0714340209961              
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 89.58584594726562                                                       
                                                                                                                                   
Epoch 3/100, Loss: 1.340375304222107, Top1 Train Accuracy: 74.10714721679688, Top5 Train Accuracy: 94.64286041259766               
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 4/100, Loss: 0.9405054450035095, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 95.53572082519531                           
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 5/100, Loss: 1.0535422563552856, Top1 Train Accuracy: 73.21428680419922, Top5 Train Accuracy: 95.53572082519531              
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 6/100, Loss: 1.2898856401443481, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 95.53572082519531                           
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 7/100, Loss: 0.912884533405304, Top1 Train Accuracy: 73.21428680419922, Top5 Train Accuracy: 95.53572082519531               
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 8/100, Loss: 1.3236007690429688, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 95.53572082519531                           
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 9/100, Loss: 0.5202780961990356, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 95.53572082519531                           
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 10/100, Loss: 1.8317428827285767, Top1 Train Accuracy: 73.21428680419922, Top5 Train Accuracy: 95.53572082519531             
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 93.00361633300781                                                       
                                                                                                                                   
Epoch 11/100, Loss: 1.0641237497329712, Top1 Train Accuracy: 74.10714721679688, Top5 Train Accuracy: 96.42857360839844             
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 12/100, Loss: 1.1086777448654175, Top1 Train Accuracy: 73.21428680419922, Top5 Train Accuracy: 98.21428680419922             
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.92320251464844                                                       
                                                                                                                                   
Epoch 13/100, Loss: 0.7460862398147583, Top1 Train Accuracy: 74.10714721679688, Top5 Train Accuracy: 97.3214340209961              
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 92.96340942382812                                                       
                                                                                                                                   
Epoch 14/100, Loss: 1.4462063312530518, Top1 Train Accuracy: 74.10714721679688, Top5 Train Accuracy: 98.21428680419922             
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 91.83755493164062                                                       
                                                                                                                                   
Epoch 15/100, Loss: 0.450213223695755, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 98.21428680419922                           
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 91.95818328857422                                                       
                                                                                                                                   
Epoch 16/100, Loss: 0.6527409553527832, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 98.21428680419922                          
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 91.39525604248047

Epoch 17/100, Loss: 0.8780703544616699, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 98.21428680419922                [215/1832]
Top1 Test Accuracy: 65.74185943603516, Top5 Test Accuracy: 91.67671966552734

Epoch 18/100, Loss: 0.6510042548179626, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 98.21428680419922
Top1 Test Accuracy: 65.299560546875, Top5 Test Accuracy: 91.71692657470703

Epoch 19/100, Loss: 0.6503241062164307, Top1 Train Accuracy: 76.78572082519531, Top5 Train Accuracy: 98.21428680419922
Top1 Test Accuracy: 65.94290161132812, Top5 Test Accuracy: 90.83232879638672

Epoch 20/100, Loss: 0.8280888199806213, Top1 Train Accuracy: 75.0, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 63.691192626953125, Top5 Test Accuracy: 91.07357788085938

Epoch 21/100, Loss: 1.1962708234786987, Top1 Train Accuracy: 76.78572082519531, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 65.78206634521484, Top5 Test Accuracy: 90.9127426147461

Epoch 22/100, Loss: 0.38647130131721497, Top1 Train Accuracy: 77.67857360839844, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 65.05830383300781, Top5 Test Accuracy: 90.39002990722656

Epoch 23/100, Loss: 0.7307423949241638, Top1 Train Accuracy: 76.78572082519531, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 64.89746856689453, Top5 Test Accuracy: 90.71170043945312

Epoch 24/100, Loss: 0.3408510386943817, Top1 Train Accuracy: 80.35714721679688, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 64.25411987304688, Top5 Test Accuracy: 90.14877319335938

Epoch 25/100, Loss: 0.5877906680107117, Top1 Train Accuracy: 81.25, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 63.530357360839844, Top5 Test Accuracy: 89.7466812133789

Epoch 26/100, Loss: 0.1043480932712555, Top1 Train Accuracy: 84.8214340209961, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 64.13349151611328, Top5 Test Accuracy: 89.666259765625

Epoch 27/100, Loss: 0.5041425228118896, Top1 Train Accuracy: 84.8214340209961, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 65.25934600830078, Top5 Test Accuracy: 88.70124816894531

Epoch 28/100, Loss: 0.27332648634910583, Top1 Train Accuracy: 90.17857360839844, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 65.37997436523438, Top5 Test Accuracy: 89.82710266113281

Epoch 29/100, Loss: 0.20284412801265717, Top1 Train Accuracy: 89.28572082519531, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 62.8468017578125, Top5 Test Accuracy: 89.06312561035156

Epoch 30/100, Loss: 0.16600185632705688, Top1 Train Accuracy: 90.17857360839844, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.55488586425781, Top5 Test Accuracy: 89.14354705810547

Epoch 31/100, Loss: 0.4359995424747467, Top1 Train Accuracy: 95.53572082519531, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 62.76638412475586, Top5 Test Accuracy: 88.5404052734375

Epoch 32/100, Loss: 0.1789553165435791, Top1 Train Accuracy: 93.75000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 64.33454132080078, Top5 Test Accuracy: 88.25894165039062

Epoch 33/100, Loss: 0.1652866005897522, Top1 Train Accuracy: 98.21428680419922, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.46923828125, Top5 Test Accuracy: 87.85685729980469

Epoch 34/100, Loss: 0.16415974497795105, Top1 Train Accuracy: 96.42857360839844, Top5 Train Accuracy: 100.00000762939453           
Top1 Test Accuracy: 60.03216552734375, Top5 Test Accuracy: 88.33936309814453                                                       

Epoch 35/100, Loss: 0.07033107429742813, Top1 Train Accuracy: 96.42857360839844, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 63.81182098388672, Top5 Test Accuracy: 89.26417541503906

Epoch 36/100, Loss: 0.07522221654653549, Top1 Train Accuracy: 97.3214340209961, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 58.624847412109375, Top5 Test Accuracy: 87.93727111816406

Epoch 37/100, Loss: 0.09956879913806915, Top1 Train Accuracy: 97.3214340209961, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 61.88178634643555, Top5 Test Accuracy: 88.09810638427734

Epoch 38/100, Loss: 0.04304195195436478, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 61.88178634643555, Top5 Test Accuracy: 88.82186889648438

Epoch 39/100, Loss: 0.08586350828409195, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 58.02171325683594, Top5 Test Accuracy: 87.21350860595703

Epoch 40/100, Loss: 0.01794620230793953, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 61.88178634643555, Top5 Test Accuracy: 86.89183807373047

Epoch 41/100, Loss: 0.0218662116676569, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 56.89585876464844, Top5 Test Accuracy: 88.25894165039062

Epoch 42/100, Loss: 0.022544337436556816, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 63.691192626953125, Top5 Test Accuracy: 86.69078826904297

Epoch 43/100, Loss: 0.02556169405579567, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.46923828125, Top5 Test Accuracy: 87.93727111816406

Epoch 44/100, Loss: 0.0037674836348742247, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.796138763427734, Top5 Test Accuracy: 87.89706420898438

Epoch 45/100, Loss: 0.0031215581111609936, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.11258316040039, Top5 Test Accuracy: 87.69601440429688

Epoch 46/100, Loss: 0.0023825836833566427, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.670284271240234, Top5 Test Accuracy: 87.21350860595703

Epoch 47/100, Loss: 0.0019636042416095734, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.83634948730469, Top5 Test Accuracy: 87.09288024902344

Epoch 48/100, Loss: 0.00692580034956336, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.47446823120117, Top5 Test Accuracy: 87.29393005371094

Epoch 49/100, Loss: 0.003568540560081601, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.41455078125

Epoch 50/100, Loss: 0.0013970196014270186, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.55488586425781, Top5 Test Accuracy: 87.37434387207031

Epoch 51/100, Loss: 0.0012720816303044558, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.83634948730469, Top5 Test Accuracy: 87.29393005371094

Epoch 52/100, Loss: 0.002725078957155347, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.37434387207031

Epoch 53/100, Loss: 0.001270015025511384, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.29393005371094

Epoch 54/100, Loss: 0.0041397446766495705, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.715721130371094, Top5 Test Accuracy: 87.13308715820312

Epoch 55/100, Loss: 0.002182489028200507, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.67551040649414, Top5 Test Accuracy: 87.29393005371094

Epoch 56/100, Loss: 0.0029025771655142307, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.35383987426758, Top5 Test Accuracy: 87.25371551513672

Epoch 57/100, Loss: 0.0015544653870165348, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.51467514038086, Top5 Test Accuracy: 87.33413696289062

Epoch 58/100, Loss: 0.002279693027958274, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.25371551513672

Epoch 59/100, Loss: 0.0027661719359457493, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.63530349731445, Top5 Test Accuracy: 87.25371551513672

Epoch 60/100, Loss: 0.00046596123138442636, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.21350860595703

Epoch 61/100, Loss: 0.0035362783819437027, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.5950927734375, Top5 Test Accuracy: 87.25371551513672

Epoch 62/100, Loss: 0.0016213224735110998, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.29393005371094

Epoch 63/100, Loss: 0.0015343361301347613, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.152793884277344, Top5 Test Accuracy: 87.25371551513672

Epoch 64/100, Loss: 0.002465813187882304, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.394046783447266, Top5 Test Accuracy: 87.29393005371094

Epoch 65/100, Loss: 0.001094754203222692, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.33413696289062

Epoch 66/100, Loss: 0.002247589873149991, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.17330169677734

Epoch 67/100, Loss: 0.0015930311055853963, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.33413696289062

Epoch 68/100, Loss: 0.0003341592091601342, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.394046783447266, Top5 Test Accuracy: 87.45476531982422

Epoch 69/100, Loss: 0.0006981560145504773, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.41455078125

Epoch 70/100, Loss: 0.0008448694716207683, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.37434387207031

Epoch 71/100, Loss: 0.0004929928691126406, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.41455078125

Epoch 72/100, Loss: 0.0020288946107029915, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.47446823120117, Top5 Test Accuracy: 87.33413696289062

Epoch 73/100, Loss: 0.0012055934639647603, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.43425750732422, Top5 Test Accuracy: 87.37434387207031

Epoch 74/100, Loss: 0.0017155089881271124, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.99195861816406, Top5 Test Accuracy: 87.29393005371094

Epoch 75/100, Loss: 0.00044856235035695136, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.99195861816406, Top5 Test Accuracy: 87.25371551513672

Epoch 76/100, Loss: 0.000951073132455349, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.25371551513672

Epoch 77/100, Loss: 0.001041320851072669, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.55488586425781, Top5 Test Accuracy: 87.33413696289062

Epoch 78/100, Loss: 0.0011068416060879827, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.233211517333984, Top5 Test Accuracy: 87.37434387207031

Epoch 79/100, Loss: 0.0005337812472134829, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.91154098510742, Top5 Test Accuracy: 87.25371551513672

Epoch 80/100, Loss: 0.0007213650387711823, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.233211517333984, Top5 Test Accuracy: 87.33413696289062

Epoch 81/100, Loss: 0.0009425837197341025, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.37434387207031

Epoch 82/100, Loss: 0.0006245817057788372, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.33413696289062

Epoch 83/100, Loss: 0.00025045263464562595, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.29393005371094

Epoch 84/100, Loss: 0.00042163656326010823, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.0723762512207, Top5 Test Accuracy: 87.25371551513672

Epoch 85/100, Loss: 0.00019084176165051758, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.1930046081543, Top5 Test Accuracy: 87.41455078125

Epoch 86/100, Loss: 0.000688510132022202, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.11258316040039, Top5 Test Accuracy: 87.41455078125

Epoch 87/100, Loss: 0.0007911680149845779, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.11258316040039, Top5 Test Accuracy: 87.37434387207031

Epoch 88/100, Loss: 0.00037640819209627807, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.33413696289062

Epoch 89/100, Loss: 0.0009809478651732206, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.35383987426758, Top5 Test Accuracy: 87.29393005371094

Epoch 90/100, Loss: 0.0004625866422429681, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.37434387207031

Epoch 91/100, Loss: 0.000598116428591311, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.99195861816406, Top5 Test Accuracy: 87.33413696289062

Epoch 92/100, Loss: 0.0005812124582007527, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.11258316040039, Top5 Test Accuracy: 87.33413696289062

Epoch 93/100, Loss: 0.0006532937404699624, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.35383987426758, Top5 Test Accuracy: 87.37434387207031

Epoch 94/100, Loss: 0.0005200410960242152, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.27342224121094, Top5 Test Accuracy: 87.25371551513672

Epoch 95/100, Loss: 0.000701002951245755, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 59.99195861816406, Top5 Test Accuracy: 87.25371551513672

Epoch 96/100, Loss: 0.0005572677473537624, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.0723762512207, Top5 Test Accuracy: 87.37434387207031

Epoch 97/100, Loss: 0.00029621238354593515, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.152793884277344, Top5 Test Accuracy: 87.41455078125

Epoch 98/100, Loss: 0.0007715225219726562, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.11258316040039, Top5 Test Accuracy: 87.41455078125

Epoch 99/100, Loss: 0.00039194634882733226, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.313629150390625, Top5 Test Accuracy: 87.37434387207031

Epoch 100/100, Loss: 0.0008221057942137122, Top1 Train Accuracy: 100.00000762939453, Top5 Train Accuracy: 100.00000762939453
Top1 Test Accuracy: 60.0723762512207, Top5 Test Accuracy: 87.33413696289062
"""

# Extract loss values using regular expression
loss_values = re.findall(r"Loss: ([-e\d.]+)", original_str)
loss_values = [float(loss) for loss in loss_values]

train_top1_acc = re.findall(r"Top1 Train Accuracy: ([\d.]+)", original_str)
train_top1_acc = [float(acc) for acc in train_top1_acc]
train_top5_acc = re.findall(r"Top5 Train Accuracy: ([\d.]+)", original_str)
train_top5_acc = [float(acc) for acc in train_top5_acc]

test_top1_acc = re.findall(r"Top1 Test Accuracy: ([\d.]+)", original_str)
test_top1_acc = [float(acc) for acc in test_top1_acc]
test_top5_acc = re.findall(r"Top5 Test Accuracy: ([\d.]+)", original_str)
test_top5_acc = [float(acc) for acc in test_top5_acc]

# assert len(loss_values) == len(train_top1_acc) == len(train_top5_acc) == len(test_top1_acc) == len(test_top5_acc) == 100

# Create a list of epochs (assuming you have 15 epochs)

# # 200 Epochs
# epochs = list(range(1, len(loss_values) + 1))
# # Plot the training loss
# plt.figure()
# plt.plot(epochs, loss_values, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss vs. Epochs')
# plt.grid()
# plt.legend()
# plt.savefig('train_loss_resnet18_supervised_200epochs.png')
# plt.show()

# # Plot the training accuracy
# plt.figure()
# plt.plot(epochs, train_acc, label='Training Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Training Accuracy')
# plt.title('Training Accuracy vs. Epochs')
# plt.grid()
# plt.legend()
# plt.savefig('train_acc_resnet18_supervised_200epochs.png')
# plt.show()

# # Plot the test accuracy
# epochs = list(range(1, 201, 20))
# plt.figure()
# plt.plot(epochs, test_acc[1:], marker='o', linestyle='-', label='Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy vs. Epochs')
# plt.grid()
# plt.legend()
# plt.savefig('test_acc_resnet18_supervised_200epochs.png')

# 100 Epochs
epochs = list(range(100))
plt.figure()
fig, ax1 = plt.subplots()

ax1.plot(epochs, loss_values, label='Training Loss', color='tab:red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(epochs, train_top1_acc, label='Training Accuracy', color='tab:blue')
ax2.set_ylabel('Top1 Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

test_epochs = list(range(0, 100, 1))
ax2.plot(test_epochs, test_top1_acc, label='Test Accuracy', color='tab:green')
ax2.legend(loc='upper right')

plt.grid()
plt.savefig(f'loss_acc_top1_001_resnet18_nine_partition_label_supervised.png')
plt.show()


plt.figure()
fig, ax1 = plt.subplots()

ax1.plot(epochs, loss_values, label='Training Loss', color='tab:red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(epochs, train_top5_acc, label='Training Accuracy', color='tab:blue')
ax2.set_ylabel('Top5 Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

test_epochs = list(range(0, 100, 1))
ax2.plot(test_epochs, test_top5_acc, label='Test Accuracy', color='tab:green')
ax2.legend(loc='upper right')

plt.grid()
plt.savefig(f'loss_acc_top5_001_resnet18_nine_partition_label_supervised.png')
plt.show()

# # Plot the test accuracy
# epochs = list(range(1, 101, 20))
# plt.figure()
# plt.plot(epochs, test_acc[1:6], marker='o', linestyle='-', label='Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy vs. Epochs')
# plt.grid()
# plt.legend()
# plt.savefig('test_acc_resnet18_supervised_100epochs.png')

# plt.show()


# 200 Epochs
epochs = list(range(len(loss_values)))
plt.figure()
fig, ax1 = plt.subplots()

ax1.plot(epochs, loss_values, label='Training Loss', color='tab:red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(epochs, train_top1_acc, label='Training Accuracy', color='tab:blue')
ax2.set_ylabel('Top1 Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='upper right')

test_epochs = list(range(0, 100))
ax2.plot(test_epochs, test_top1_acc, label='Test Accuracy', color='tab:green')
ax2.legend(loc='upper right')

plt.grid()
plt.savefig('loss_acc_top1_001_resnet18_nine_partition_label_supervised.png')
plt.show()
