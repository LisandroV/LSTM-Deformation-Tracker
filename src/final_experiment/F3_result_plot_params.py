"""
   Plot the params result of the random search with teacher forcing.
"""
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

experiment_1 = [{'rating': 0, 'name': 'Adam', 'learning_rate': 0.0036187424336578482, 'decay': 0.0, 'beta_1': 0.7363340404583201, 'beta_2': 0.999, 'epsilon': 6.523462128282289e-07, 'amsgrad': False}, {'rating': 1, 'name': 'Adam', 'learning_rate': 0.0011235232517236293, 'decay': 0.0, 'beta_1': 0.7003775431950426, 'beta_2': 0.999, 'epsilon': 1.561120169832853e-07, 'amsgrad': False}, {'rating': 2, 'name': 'Adam', 'learning_rate': 0.004498016954720033, 'decay': 0.0, 'beta_1': 0.820311658518327, 'beta_2': 0.999, 'epsilon': 6.716926024984373e-07, 'amsgrad': False}, {'rating': 3, 'name': 'Adam', 'learning_rate': 0.003989554434400411, 'decay': 0.0, 'beta_1': 0.7926391602740768, 'beta_2': 0.999, 'epsilon': 2.823872825436216e-07, 'amsgrad': False}, {'rating': 4, 'name': 'Adam', 'learning_rate': 0.0034619351036427022, 'decay': 0.0, 'beta_1': 0.8456471098457944, 'beta_2': 0.999, 'epsilon': 3.2909470862044774e-07, 'amsgrad': False}, {'rating': 5, 'name': 'Adam', 'learning_rate': 0.0045399976933986385, 'decay': 0.0, 'beta_1': 0.8299444854733801, 'beta_2': 0.999, 'epsilon': 3.8556514639518343e-07, 'amsgrad': False}, {'rating': 6, 'name': 'Adam', 'learning_rate': 0.0035748093641514225, 'decay': 0.0, 'beta_1': 0.7993722192453643, 'beta_2': 0.999, 'epsilon': 1.3994416677080426e-07, 'amsgrad': False}, {'rating': 7, 'name': 'Adam', 'learning_rate': 0.0019493670232946807, 'decay': 0.0, 'beta_1': 0.7186984392182776, 'beta_2': 0.999, 'epsilon': 1.1770879859440486e-07, 'amsgrad': False}, {'rating': 8, 'name': 'Adam', 'learning_rate': 0.00269999761557139, 'decay': 0.0, 'beta_1': 0.7225849635673374, 'beta_2': 0.999, 'epsilon': 8.196774218816485e-07, 'amsgrad': False}, {'rating': 9, 'name': 'Adam', 'learning_rate': 0.002548049781360705, 'decay': 0.0, 'beta_1': 0.9447625300181935, 'beta_2': 0.999, 'epsilon': 1.4066019216223632e-07, 'amsgrad': False}, {'rating': 10, 'name': 'Adam', 'learning_rate': 0.0014869059400314665, 'decay': 0.0, 'beta_1': 0.769895759651467, 'beta_2': 0.999, 'epsilon': 1.3505797378360447e-07, 'amsgrad': False}, {'rating': 11, 'name': 'Adam', 'learning_rate': 0.001683813074959655, 'decay': 0.0, 'beta_1': 0.769953868393247, 'beta_2': 0.999, 'epsilon': 1.184204357462253e-07, 'amsgrad': False}, {'rating': 12, 'name': 'Adam', 'learning_rate': 0.0033988349215458057, 'decay': 0.0, 'beta_1': 0.7486343119832952, 'beta_2': 0.999, 'epsilon': 2.3474389703503775e-07, 'amsgrad': False}, {'rating': 13, 'name': 'Adam', 'learning_rate': 0.002272414950384994, 'decay': 0.0, 'beta_1': 0.9401951891888081, 'beta_2': 0.999, 'epsilon': 1.1281415605346916e-07, 'amsgrad': False}, {'rating': 14, 'name': 'Adam', 'learning_rate': 0.0014344946151797695, 'decay': 0.0, 'beta_1': 0.8236218589948586, 'beta_2': 0.999, 'epsilon': 2.0314528870693706e-07, 'amsgrad': False}, {'rating': 15, 'name': 'Adam', 'learning_rate': 0.0016614211520049177, 'decay': 0.0, 'beta_1': 0.823458162610393, 'beta_2': 0.999, 'epsilon': 2.584811202621318e-07, 'amsgrad': False}, {'rating': 16, 'name': 'Adam', 'learning_rate': 0.0019692350445046107, 'decay': 0.0, 'beta_1': 0.8367003883841582, 'beta_2': 0.999, 'epsilon': 8.172282282086946e-07, 'amsgrad': False}, {'rating': 17, 'name': 'Adam', 'learning_rate': 0.004733131156232761, 'decay': 0.0, 'beta_1': 0.8451892912334891, 'beta_2': 0.999, 'epsilon': 1.1029751136188258e-07, 'amsgrad': False}, {'rating': 18, 'name': 'Adam', 'learning_rate': 0.0044408377152642935, 'decay': 0.0, 'beta_1': 0.7218253717164541, 'beta_2': 0.999, 'epsilon': 3.272939323057123e-07, 'amsgrad': False}, {'rating': 19, 'name': 'Adam', 'learning_rate': 0.0017746456796230426, 'decay': 0.0, 'beta_1': 0.7364173137760326, 'beta_2': 0.999, 'epsilon': 2.5350006495141605e-07, 'amsgrad': False}, {'rating': 20, 'name': 'Adam', 'learning_rate': 0.0037506304684171805, 'decay': 0.0, 'beta_1': 0.877745566949071, 'beta_2': 0.999, 'epsilon': 1.738213145694833e-07, 'amsgrad': False}, {'rating': 21, 'name': 'Adam', 'learning_rate': 0.0015479757204326444, 'decay': 0.0, 'beta_1': 0.7361801381584171, 'beta_2': 0.999, 'epsilon': 3.2058294469501294e-07, 'amsgrad': False}, {'rating': 22, 'name': 'Adam', 'learning_rate': 0.0011394097782884429, 'decay': 0.0, 'beta_1': 0.7208801302883964, 'beta_2': 0.999, 'epsilon': 1.5983447078296182e-07, 'amsgrad': False}, {'rating': 23, 'name': 'Adam', 'learning_rate': 0.0021796385932708385, 'decay': 0.0, 'beta_1': 0.8439290520341076, 'beta_2': 0.999, 'epsilon': 4.672346432478633e-07, 'amsgrad': False}, {'rating': 24, 'name': 'Adam', 'learning_rate': 0.004452852083606927, 'decay': 0.0, 'beta_1': 0.8190395655573364, 'beta_2': 0.999, 'epsilon': 2.787014480323111e-07, 'amsgrad': False}, {'rating': 25, 'name': 'Adam', 'learning_rate': 0.00425709848453973, 'decay': 0.0, 'beta_1': 0.7111439487195851, 'beta_2': 0.999, 'epsilon': 3.7374616007062716e-07, 'amsgrad': False}, {'rating': 26, 'name': 'Adam', 'learning_rate': 0.0010504061414228025, 'decay': 0.0, 'beta_1': 0.9239755937517541, 'beta_2': 0.999, 'epsilon': 1.7415482098502808e-07, 'amsgrad': False}, {'rating': 27, 'name': 'Adam', 'learning_rate': 0.00468795997982379, 'decay': 0.0, 'beta_1': 0.8776709332656232, 'beta_2': 0.999, 'epsilon': 7.552649820976935e-07, 'amsgrad': False}, {'rating': 28, 'name': 'Adam', 'learning_rate': 0.002222277453902146, 'decay': 0.0, 'beta_1': 0.749897176588525, 'beta_2': 0.999, 'epsilon': 5.389284745949776e-07, 'amsgrad': False}, {'rating': 29, 'name': 'Adam', 'learning_rate': 0.003299167300511539, 'decay': 0.0, 'beta_1': 0.927442309480968, 'beta_2': 0.999, 'epsilon': 4.833268839125031e-07, 'amsgrad': False}]
experiment_2 = [{'rating': 0, 'name': 'Adam', 'learning_rate': 0.001078203282079985, 'decay': 0.0, 'beta_1': 0.9136968841686723, 'beta_2': 0.999, 'epsilon': 8.652341013580072e-06, 'amsgrad': False}, {'rating': 1, 'name': 'Adam', 'learning_rate': 0.004389609210070219, 'decay': 0.0, 'beta_1': 0.9437139377620483, 'beta_2': 0.999, 'epsilon': 1.1760954202181643e-06, 'amsgrad': False}, {'rating': 2, 'name': 'Adam', 'learning_rate': 0.003099089425714443, 'decay': 0.0, 'beta_1': 0.816232223476276, 'beta_2': 0.999, 'epsilon': 8.983918613255643e-06, 'amsgrad': False}, {'rating': 3, 'name': 'Adam', 'learning_rate': 0.001062474184478115, 'decay': 0.0, 'beta_1': 0.9220338833310469, 'beta_2': 0.999, 'epsilon': 1.395893537440065e-06, 'amsgrad': False}, {'rating': 4, 'name': 'Adam', 'learning_rate': 0.0015206186091342982, 'decay': 0.0, 'beta_1': 0.7978158630481034, 'beta_2': 0.999, 'epsilon': 5.389430311620116e-06, 'amsgrad': False}, {'rating': 5, 'name': 'Adam', 'learning_rate': 0.002519263456746685, 'decay': 0.0, 'beta_1': 0.8260963196672702, 'beta_2': 0.999, 'epsilon': 1.987731677846987e-06, 'amsgrad': False}, {'rating': 6, 'name': 'Adam', 'learning_rate': 0.0021599680299972846, 'decay': 0.0, 'beta_1': 0.9229973957604414, 'beta_2': 0.999, 'epsilon': 1.1922343682920234e-06, 'amsgrad': False}, {'rating': 7, 'name': 'Adam', 'learning_rate': 0.0012445908435230314, 'decay': 0.0, 'beta_1': 0.9177777879149707, 'beta_2': 0.999, 'epsilon': 8.506993746480856e-06, 'amsgrad': False}, {'rating': 8, 'name': 'Adam', 'learning_rate': 0.001281303241718214, 'decay': 0.0, 'beta_1': 0.8295431321293527, 'beta_2': 0.999, 'epsilon': 1.1313488875509762e-06, 'amsgrad': False}, {'rating': 9, 'name': 'Adam', 'learning_rate': 0.001861401947777535, 'decay': 0.0, 'beta_1': 0.8057865255181482, 'beta_2': 0.999, 'epsilon': 8.685222678115228e-06, 'amsgrad': False}, {'rating': 10, 'name': 'Adam', 'learning_rate': 0.00260844431190988, 'decay': 0.0, 'beta_1': 0.7085878360958512, 'beta_2': 0.999, 'epsilon': 8.03857595558115e-06, 'amsgrad': False}, {'rating': 11, 'name': 'Adam', 'learning_rate': 0.001458809927716741, 'decay': 0.0, 'beta_1': 0.7420949058767332, 'beta_2': 0.999, 'epsilon': 2.502624736906155e-06, 'amsgrad': False}, {'rating': 12, 'name': 'Adam', 'learning_rate': 0.0012393039621076043, 'decay': 0.0, 'beta_1': 0.8854492268126183, 'beta_2': 0.999, 'epsilon': 2.268185899458389e-06, 'amsgrad': False}, {'rating': 13, 'name': 'Adam', 'learning_rate': 0.004363182836070574, 'decay': 0.0, 'beta_1': 0.787117997482079, 'beta_2': 0.999, 'epsilon': 4.022486539561283e-06, 'amsgrad': False}, {'rating': 14, 'name': 'Adam', 'learning_rate': 0.0012781587444886215, 'decay': 0.0, 'beta_1': 0.7507341359006839, 'beta_2': 0.999, 'epsilon': 2.480177192299634e-06, 'amsgrad': False}, {'rating': 15, 'name': 'Adam', 'learning_rate': 0.002783448726510062, 'decay': 0.0, 'beta_1': 0.8252618931146327, 'beta_2': 0.999, 'epsilon': 2.858323547652605e-06, 'amsgrad': False}, {'rating': 16, 'name': 'Adam', 'learning_rate': 0.0028496481835064134, 'decay': 0.0, 'beta_1': 0.8444251841929494, 'beta_2': 0.999, 'epsilon': 1.5407246639409265e-06, 'amsgrad': False}, {'rating': 17, 'name': 'Adam', 'learning_rate': 0.0025623499361644734, 'decay': 0.0, 'beta_1': 0.7591321006826128, 'beta_2': 0.999, 'epsilon': 3.52867986190572e-06, 'amsgrad': False}, {'rating': 18, 'name': 'Adam', 'learning_rate': 0.004302710516892558, 'decay': 0.0, 'beta_1': 0.7136293827432176, 'beta_2': 0.999, 'epsilon': 1.002921248273213e-06, 'amsgrad': False}, {'rating': 19, 'name': 'Adam', 'learning_rate': 0.0033105405884398574, 'decay': 0.0, 'beta_1': 0.9161340999065354, 'beta_2': 0.999, 'epsilon': 9.764279192758044e-06, 'amsgrad': False}, {'rating': 20, 'name': 'Adam', 'learning_rate': 0.0017791193776547207, 'decay': 0.0, 'beta_1': 0.8511881844005849, 'beta_2': 0.999, 'epsilon': 1.938236981634261e-06, 'amsgrad': False}, {'rating': 21, 'name': 'Adam', 'learning_rate': 0.002127330474761465, 'decay': 0.0, 'beta_1': 0.8783326580117019, 'beta_2': 0.999, 'epsilon': 1.3714085784156307e-06, 'amsgrad': False}, {'rating': 22, 'name': 'Adam', 'learning_rate': 0.004227245174565477, 'decay': 0.0, 'beta_1': 0.8513877500692394, 'beta_2': 0.999, 'epsilon': 1.6264521820405237e-06, 'amsgrad': False}, {'rating': 23, 'name': 'Adam', 'learning_rate': 0.003535975407030089, 'decay': 0.0, 'beta_1': 0.7836256042240343, 'beta_2': 0.999, 'epsilon': 8.267520698231623e-06, 'amsgrad': False}, {'rating': 24, 'name': 'Adam', 'learning_rate': 0.0020311899870553807, 'decay': 0.0, 'beta_1': 0.8265057087576262, 'beta_2': 0.999, 'epsilon': 9.152325246234564e-06, 'amsgrad': False}, {'rating': 25, 'name': 'Adam', 'learning_rate': 0.0011405354347752646, 'decay': 0.0, 'beta_1': 0.7227663069099494, 'beta_2': 0.999, 'epsilon': 2.872628714495347e-06, 'amsgrad': False}, {'rating': 26, 'name': 'Adam', 'learning_rate': 0.0021763568708372412, 'decay': 0.0, 'beta_1': 0.8743782656424429, 'beta_2': 0.999, 'epsilon': 3.3532883033423103e-06, 'amsgrad': False}, {'rating': 27, 'name': 'Adam', 'learning_rate': 0.0033061942498295166, 'decay': 0.0, 'beta_1': 0.8273012134818165, 'beta_2': 0.999, 'epsilon': 6.791493549551119e-06, 'amsgrad': False}, {'rating': 28, 'name': 'Adam', 'learning_rate': 0.0034875281248137116, 'decay': 0.0, 'beta_1': 0.8521681938551895, 'beta_2': 0.999, 'epsilon': 1.4606164689667613e-06, 'amsgrad': False}, {'rating': 29, 'name': 'Adam', 'learning_rate': 0.003025201550151962, 'decay': 0.0, 'beta_1': 0.9211427366956799, 'beta_2': 0.999, 'epsilon': 2.1985296116741494e-06, 'amsgrad': False}]
experiment_3 = [{'rating': 0, 'name': 'Adam', 'learning_rate': 0.006923375185711218, 'decay': 0.0, 'beta_1': 0.8442654629126807, 'beta_2': 0.999, 'epsilon': 1.984571030380454e-07, 'amsgrad': False}, {'rating': 1, 'name': 'Adam', 'learning_rate': 0.006345862044688259, 'decay': 0.0, 'beta_1': 0.7458588760464884, 'beta_2': 0.999, 'epsilon': 1.2749613877742427e-07, 'amsgrad': False}, {'rating': 2, 'name': 'Adam', 'learning_rate': 0.008916370246589792, 'decay': 0.0, 'beta_1': 0.8538515782936682, 'beta_2': 0.999, 'epsilon': 3.8274464228869826e-07, 'amsgrad': False}, {'rating': 3, 'name': 'Adam', 'learning_rate': 0.006533120833007976, 'decay': 0.0, 'beta_1': 0.7597881983651766, 'beta_2': 0.999, 'epsilon': 4.4802643329892343e-07, 'amsgrad': False}, {'rating': 4, 'name': 'Adam', 'learning_rate': 0.006480448609545671, 'decay': 0.0, 'beta_1': 0.9355025971742776, 'beta_2': 0.999, 'epsilon': 7.435740607050862e-07, 'amsgrad': False}, {'rating': 5, 'name': 'Adam', 'learning_rate': 0.006239367203777412, 'decay': 0.0, 'beta_1': 0.7354974407756975, 'beta_2': 0.999, 'epsilon': 3.236388963604789e-07, 'amsgrad': False}, {'rating': 6, 'name': 'Adam', 'learning_rate': 0.006114614547299648, 'decay': 0.0, 'beta_1': 0.9102152796920815, 'beta_2': 0.999, 'epsilon': 3.1273367531538505e-07, 'amsgrad': False}, {'rating': 7, 'name': 'Adam', 'learning_rate': 0.005291237753935805, 'decay': 0.0, 'beta_1': 0.899181331367268, 'beta_2': 0.999, 'epsilon': 5.88211156998908e-07, 'amsgrad': False}, {'rating': 8, 'name': 'Adam', 'learning_rate': 0.00999284955468417, 'decay': 0.0, 'beta_1': 0.7188508120989483, 'beta_2': 0.999, 'epsilon': 6.84624553643233e-07, 'amsgrad': False}, {'rating': 9, 'name': 'Adam', 'learning_rate': 0.009213132647106437, 'decay': 0.0, 'beta_1': 0.8325879178950081, 'beta_2': 0.999, 'epsilon': 8.699961988209932e-07, 'amsgrad': False}, {'rating': 10, 'name': 'Adam', 'learning_rate': 0.007233577072531344, 'decay': 0.0, 'beta_1': 0.7644771237107164, 'beta_2': 0.999, 'epsilon': 4.802311358080774e-07, 'amsgrad': False}, {'rating': 11, 'name': 'Adam', 'learning_rate': 0.008387538002086086, 'decay': 0.0, 'beta_1': 0.7887733507918501, 'beta_2': 0.999, 'epsilon': 1.7568639420117054e-07, 'amsgrad': False}, {'rating': 12, 'name': 'Adam', 'learning_rate': 0.005873154834354088, 'decay': 0.0, 'beta_1': 0.7081152263237671, 'beta_2': 0.999, 'epsilon': 5.217279525779011e-07, 'amsgrad': False}, {'rating': 13, 'name': 'Adam', 'learning_rate': 0.007618131597724829, 'decay': 0.0, 'beta_1': 0.9243768715606387, 'beta_2': 0.999, 'epsilon': 2.3909066715992934e-07, 'amsgrad': False}, {'rating': 14, 'name': 'Adam', 'learning_rate': 0.007555272748033133, 'decay': 0.0, 'beta_1': 0.809803452533589, 'beta_2': 0.999, 'epsilon': 1.8785529787673334e-07, 'amsgrad': False}, {'rating': 15, 'name': 'Adam', 'learning_rate': 0.005986084705781783, 'decay': 0.0, 'beta_1': 0.8705744062757242, 'beta_2': 0.999, 'epsilon': 6.661726250337766e-07, 'amsgrad': False}, {'rating': 16, 'name': 'Adam', 'learning_rate': 0.0054264887199665015, 'decay': 0.0, 'beta_1': 0.7080052649845634, 'beta_2': 0.999, 'epsilon': 5.506326212551811e-07, 'amsgrad': False}, {'rating': 17, 'name': 'Adam', 'learning_rate': 0.005526867090356502, 'decay': 0.0, 'beta_1': 0.9226191316145441, 'beta_2': 0.999, 'epsilon': 1.4006121923126723e-07, 'amsgrad': False}, {'rating': 18, 'name': 'Adam', 'learning_rate': 0.009738780212317688, 'decay': 0.0, 'beta_1': 0.9195891619686495, 'beta_2': 0.999, 'epsilon': 6.87903122162237e-07, 'amsgrad': False}, {'rating': 19, 'name': 'Adam', 'learning_rate': 0.007720892324339871, 'decay': 0.0, 'beta_1': 0.735842443879938, 'beta_2': 0.999, 'epsilon': 1.7393044084091192e-07, 'amsgrad': False}, {'rating': 20, 'name': 'Adam', 'learning_rate': 0.007020885074570773, 'decay': 0.0, 'beta_1': 0.8772212909014316, 'beta_2': 0.999, 'epsilon': 1.864677984257327e-07, 'amsgrad': False}, {'rating': 21, 'name': 'Adam', 'learning_rate': 0.009401233701302861, 'decay': 0.0, 'beta_1': 0.9328913986068748, 'beta_2': 0.999, 'epsilon': 1.4474588394453585e-07, 'amsgrad': False}, {'rating': 22, 'name': 'Adam', 'learning_rate': 0.007897271846603795, 'decay': 0.0, 'beta_1': 0.9309792769970486, 'beta_2': 0.999, 'epsilon': 8.474367770808754e-07, 'amsgrad': False}, {'rating': 23, 'name': 'Adam', 'learning_rate': 0.009631636853211795, 'decay': 0.0, 'beta_1': 0.7230882509728879, 'beta_2': 0.999, 'epsilon': 7.077362632896225e-07, 'amsgrad': False}, {'rating': 24, 'name': 'Adam', 'learning_rate': 0.009124727462199207, 'decay': 0.0, 'beta_1': 0.7120855843094972, 'beta_2': 0.999, 'epsilon': 1.0050183464915359e-07, 'amsgrad': False}, {'rating': 25, 'name': 'Adam', 'learning_rate': 0.007897567320461674, 'decay': 0.0, 'beta_1': 0.9065414500170065, 'beta_2': 0.999, 'epsilon': 4.266514253707781e-07, 'amsgrad': False}, {'rating': 26, 'name': 'Adam', 'learning_rate': 0.005150112304064388, 'decay': 0.0, 'beta_1': 0.9297491030985774, 'beta_2': 0.999, 'epsilon': 1.403702355311312e-07, 'amsgrad': False}, {'rating': 27, 'name': 'Adam', 'learning_rate': 0.007776378646904655, 'decay': 0.0, 'beta_1': 0.8961913525245432, 'beta_2': 0.999, 'epsilon': 1.5468387005235862e-07, 'amsgrad': False}, {'rating': 28, 'name': 'Adam', 'learning_rate': 0.007186953292738472, 'decay': 0.0, 'beta_1': 0.7391401008310784, 'beta_2': 0.999, 'epsilon': 9.334753053365586e-07, 'amsgrad': False}, {'rating': 29, 'name': 'Adam', 'learning_rate': 0.006309897733105381, 'decay': 0.0, 'beta_1': 0.7990024333120831, 'beta_2': 0.999, 'epsilon': 2.0610383591243447e-07, 'amsgrad': False}]
experiment_4 = [{'rating': 0, 'name': 'Adam', 'learning_rate': 0.009578077231644088, 'decay': 0.0, 'beta_1': 0.8657329882043239, 'beta_2': 0.999, 'epsilon': 2.801290554825321e-06, 'amsgrad': False}, {'rating': 1, 'name': 'Adam', 'learning_rate': 0.00959883917289962, 'decay': 0.0, 'beta_1': 0.883027628878419, 'beta_2': 0.999, 'epsilon': 3.3296355583287185e-06, 'amsgrad': False}, {'rating': 2, 'name': 'Adam', 'learning_rate': 0.007039662952024265, 'decay': 0.0, 'beta_1': 0.7860603176187391, 'beta_2': 0.999, 'epsilon': 3.4010608652574186e-06, 'amsgrad': False}, {'rating': 3, 'name': 'Adam', 'learning_rate': 0.005062812669939188, 'decay': 0.0, 'beta_1': 0.8006590352548375, 'beta_2': 0.999, 'epsilon': 5.640252917281334e-06, 'amsgrad': False}, {'rating': 4, 'name': 'Adam', 'learning_rate': 0.005925384359950263, 'decay': 0.0, 'beta_1': 0.8805853106814601, 'beta_2': 0.999, 'epsilon': 1.0373203528599616e-06, 'amsgrad': False}, {'rating': 5, 'name': 'Adam', 'learning_rate': 0.008669478283726398, 'decay': 0.0, 'beta_1': 0.7781200768031261, 'beta_2': 0.999, 'epsilon': 9.951809875014272e-06, 'amsgrad': False}, {'rating': 6, 'name': 'Adam', 'learning_rate': 0.005585291941693512, 'decay': 0.0, 'beta_1': 0.8634692444262059, 'beta_2': 0.999, 'epsilon': 2.9153144375543712e-06, 'amsgrad': False}, {'rating': 7, 'name': 'Adam', 'learning_rate': 0.00680825095507282, 'decay': 0.0, 'beta_1': 0.8405847262399777, 'beta_2': 0.999, 'epsilon': 1.1294836042017014e-06, 'amsgrad': False}, {'rating': 8, 'name': 'Adam', 'learning_rate': 0.009303394878977092, 'decay': 0.0, 'beta_1': 0.7765349415226285, 'beta_2': 0.999, 'epsilon': 8.100329127584746e-06, 'amsgrad': False}, {'rating': 9, 'name': 'Adam', 'learning_rate': 0.008933994170517153, 'decay': 0.0, 'beta_1': 0.8350952214169249, 'beta_2': 0.999, 'epsilon': 1.3236325964295175e-06, 'amsgrad': False}, {'rating': 10, 'name': 'Adam', 'learning_rate': 0.008654554152415166, 'decay': 0.0, 'beta_1': 0.8240576870747094, 'beta_2': 0.999, 'epsilon': 8.372110329649253e-06, 'amsgrad': False}, {'rating': 11, 'name': 'Adam', 'learning_rate': 0.006421350784775261, 'decay': 0.0, 'beta_1': 0.8090801374854797, 'beta_2': 0.999, 'epsilon': 1.2709051957700377e-06, 'amsgrad': False}, {'rating': 12, 'name': 'Adam', 'learning_rate': 0.005132074904533007, 'decay': 0.0, 'beta_1': 0.7823452807071147, 'beta_2': 0.999, 'epsilon': 8.053171290378073e-06, 'amsgrad': False}, {'rating': 13, 'name': 'Adam', 'learning_rate': 0.008767054558434092, 'decay': 0.0, 'beta_1': 0.9322643189018276, 'beta_2': 0.999, 'epsilon': 2.166141654195157e-06, 'amsgrad': False}, {'rating': 14, 'name': 'Adam', 'learning_rate': 0.00888958440470484, 'decay': 0.0, 'beta_1': 0.9089689028050534, 'beta_2': 0.999, 'epsilon': 6.856102771446266e-06, 'amsgrad': False}, {'rating': 15, 'name': 'Adam', 'learning_rate': 0.007572705113382725, 'decay': 0.0, 'beta_1': 0.9098176261828503, 'beta_2': 0.999, 'epsilon': 1.949394816788712e-06, 'amsgrad': False}, {'rating': 16, 'name': 'Adam', 'learning_rate': 0.008759280753310037, 'decay': 0.0, 'beta_1': 0.7075123745875073, 'beta_2': 0.999, 'epsilon': 9.967827887583569e-06, 'amsgrad': False}, {'rating': 17, 'name': 'Adam', 'learning_rate': 0.009847711555699119, 'decay': 0.0, 'beta_1': 0.872496174402993, 'beta_2': 0.999, 'epsilon': 9.312683308024615e-06, 'amsgrad': False}, {'rating': 18, 'name': 'Adam', 'learning_rate': 0.008129770413782104, 'decay': 0.0, 'beta_1': 0.7465015767980214, 'beta_2': 0.999, 'epsilon': 5.894898673696617e-06, 'amsgrad': False}, {'rating': 19, 'name': 'Adam', 'learning_rate': 0.008281869877872779, 'decay': 0.0, 'beta_1': 0.8608141778906017, 'beta_2': 0.999, 'epsilon': 7.792268273550048e-06, 'amsgrad': False}, {'rating': 20, 'name': 'Adam', 'learning_rate': 0.005701241379381278, 'decay': 0.0, 'beta_1': 0.8548970291935536, 'beta_2': 0.999, 'epsilon': 1.3947906929516782e-06, 'amsgrad': False}, {'rating': 21, 'name': 'Adam', 'learning_rate': 0.009428626975797216, 'decay': 0.0, 'beta_1': 0.7576376397055431, 'beta_2': 0.999, 'epsilon': 8.703741589055233e-06, 'amsgrad': False}, {'rating': 22, 'name': 'Adam', 'learning_rate': 0.009105542568534991, 'decay': 0.0, 'beta_1': 0.9296354910740006, 'beta_2': 0.999, 'epsilon': 4.5552680069664625e-06, 'amsgrad': False}, {'rating': 23, 'name': 'Adam', 'learning_rate': 0.00789681247839697, 'decay': 0.0, 'beta_1': 0.7636882296792311, 'beta_2': 0.999, 'epsilon': 2.5335780685998957e-06, 'amsgrad': False}, {'rating': 24, 'name': 'Adam', 'learning_rate': 0.009791466566198918, 'decay': 0.0, 'beta_1': 0.7465902371949014, 'beta_2': 0.999, 'epsilon': 1.6219733029206e-06, 'amsgrad': False}, {'rating': 25, 'name': 'Adam', 'learning_rate': 0.009623114933241915, 'decay': 0.0, 'beta_1': 0.8750861720179743, 'beta_2': 0.999, 'epsilon': 7.163642271778745e-06, 'amsgrad': False}, {'rating': 26, 'name': 'Adam', 'learning_rate': 0.007807745637807979, 'decay': 0.0, 'beta_1': 0.7620463180657622, 'beta_2': 0.999, 'epsilon': 3.783472961615264e-06, 'amsgrad': False}, {'rating': 27, 'name': 'Adam', 'learning_rate': 0.007934506989895323, 'decay': 0.0, 'beta_1': 0.9152855647403378, 'beta_2': 0.999, 'epsilon': 4.5739219958423715e-06, 'amsgrad': False}, {'rating': 28, 'name': 'Adam', 'learning_rate': 0.006369335466094464, 'decay': 0.0, 'beta_1': 0.9082131930349262, 'beta_2': 0.999, 'epsilon': 1.4007385958384696e-06, 'amsgrad': False}, {'rating': 29, 'name': 'Adam', 'learning_rate': 0.007650151392204442, 'decay': 0.0, 'beta_1': 0.8549051316762469, 'beta_2': 0.999, 'epsilon': 1.0749195207340762e-06, 'amsgrad': False}]

ratings = [
    *experiment_1,
    *experiment_2,
    *experiment_3,
    *experiment_4,
]


# epsilon  X  learning_rate    ---------------------------------------------
fig = plt.figure("Hiperparámetros en la búsqueda aleatoria")
fig.suptitle("Hiperparámetros en la búsqueda aleatoria")
ax = fig.add_subplot(111)

x = np.array([r['epsilon'] for r in ratings])
y = np.array([r['learning_rate'] for r in ratings])

color_rating = np.array([1-(30-r['rating'])/30 for r in ratings])
area = np.array([ 300 if ratings[i]['rating'] < 2 else 15 for i in range(120)])


ax.scatter(x, y,c=color_rating, cmap='cool', s=area)

ax.set_xlabel('epsilon')
ax.set_ylabel('learning_rate')
plt.yscale("log")
plt.xscale("log")
plt.show()


# # beta_1  X  learning_rate    ---------------------------------------------
fig = plt.figure("Hiperparámetros en la búsqueda aleatoria")
fig.suptitle("Hiperparámetros en la búsqueda aleatoria")
ax = fig.add_subplot(111)

x = np.array([r['beta_1'] for r in ratings])
y = np.array([r['learning_rate'] for r in ratings])

color_rating = np.array([1-(120-r['rating'])/120 for r in ratings])
ax.scatter(x, y,c=color_rating, cmap='cool', s=area)

ax.set_xlabel('beta_1')
ax.set_ylabel('learning_rate')
plt.yscale("log")
plt.show()


# Showing 10 best trials Experiment 1 ------------------------------------------------------
# Trial 30 Complete [00h 52m 33s]
# val_loss: 0.004876380320638418
# Best val_loss So Far: 0.004275517072528601
# Total elapsed time: 1d 00h 54m 05s
#Search space summary
lr = {'default': 0.001, 'conditions': [], 'min_value': 0.001, 'max_value': 0.005, 'step': None, 'sampling': 'log'}
epsilon ={'default': 1e-07, 'conditions': [], 'min_value': 1e-07, 'max_value': 1e-06, 'step': None, 'sampling': 'log'}
beta_1 = {'default': 0.7, 'conditions': [], 'min_value': 0.7, 'max_value': 0.95, 'step': None, 'sampling': 'linear'}
# Trial 08 summary
# Hyperparameters:
# lr: 0.0036187424336578482
# epsilon: 6.523462128282289e-07
# beta_1: 0.7363340404583201
# Score: 0.004275517072528601

# Trial 28 summary
# Hyperparameters:
# lr: 0.0011235232517236293
# epsilon: 1.561120169832853e-07
# beta_1: 0.7003775431950426
# Score: 0.004713442642241716

# Trial 19 summary
# Hyperparameters:
# lr: 0.004498016954720033
# epsilon: 6.716926024984373e-07
# beta_1: 0.820311658518327
# Score: 0.004765646066516638

# Trial 07 summary
# Hyperparameters:
# lr: 0.003989554434400411
# epsilon: 2.823872825436216e-07
# beta_1: 0.7926391602740768
# Score: 0.004867597483098507

# Trial 29 summary
# Hyperparameters:
# lr: 0.0034619351036427022
# epsilon: 3.2909470862044774e-07
# beta_1: 0.8456471098457944
# Score: 0.004876380320638418

# Trial 25 summary
# Hyperparameters:
# lr: 0.0045399976933986385
# epsilon: 3.8556514639518343e-07
# beta_1: 0.8299444854733801
# Score: 0.005070723593235016

# Trial 11 summary
# Hyperparameters:
# lr: 0.0035748093641514225
# epsilon: 1.3994416677080426e-07
# beta_1: 0.7993722192453643
# Score: 0.005263712722808123

# Trial 04 summary
# Hyperparameters:
# lr: 0.0019493670232946807
# epsilon: 1.1770879859440486e-07
# beta_1: 0.7186984392182776
# Score: 0.005282887723296881

# Trial 00 summary
# Hyperparameters:
# lr: 0.00269999761557139
# epsilon: 8.196774218816485e-07
# beta_1: 0.7225849635673374
# Score: 0.005351308733224869

# Trial 16 summary
# Hyperparameters:
# lr: 0.002548049781360705
# epsilon: 1.4066019216223632e-07
# beta_1: 0.9447625300181935
# Score: 0.005482915323227644



# Showing 10 best trials Experiment 2 ------------------------------------------------------
# Trial 30 Complete [00h 52m 38s]
# val_loss: 0.006221266463398933
# Best val_loss So Far: 0.004057165235280991
# Total elapsed time: 1d 00h 57m 09s
#Search space summary
lr = {'default': 0.001, 'conditions': [], 'min_value': 0.001, 'max_value': 0.005, 'step': None, 'sampling': 'log'}
epsilon = {'default': 1e-06, 'conditions': [], 'min_value': 1e-06, 'max_value': 1e-05, 'step': None, 'sampling': 'log'}
beta_1 = {'default': 0.7, 'conditions': [], 'min_value': 0.7, 'max_value': 0.95, 'step': None, 'sampling': 'linear'}
# Trial 12 summary
# Hyperparameters:
# lr: 0.001078203282079985
# epsilon: 8.652341013580072e-06
# beta_1: 0.9136968841686723
# Score: 0.004057165235280991

# Trial 03 summary
# Hyperparameters:
# lr: 0.004389609210070219
# epsilon: 1.1760954202181643e-06
# beta_1: 0.9437139377620483
# Score: 0.004621021915227175

# Trial 25 summary
# Hyperparameters:
# lr: 0.003099089425714443
# epsilon: 8.983918613255643e-06
# beta_1: 0.816232223476276
# Score: 0.00493389368057251

# Trial 23 summary
# Hyperparameters:
# lr: 0.001062474184478115
# epsilon: 1.395893537440065e-06
# beta_1: 0.9220338833310469
# Score: 0.005206343252211809

# Trial 09 summary
# Hyperparameters:
# lr: 0.0015206186091342982
# epsilon: 5.389430311620116e-06
# beta_1: 0.7978158630481034
# Score: 0.005220781080424786

# Trial 01 summary
# Hyperparameters:
# lr: 0.002519263456746685
# epsilon: 1.987731677846987e-06
# beta_1: 0.8260963196672702
# Score: 0.005309212952852249

# Trial 10 summary
# Hyperparameters:
# lr: 0.0021599680299972846
# epsilon: 1.1922343682920234e-06
# beta_1: 0.9229973957604414
# Score: 0.005310652777552605

# Trial 13 summary
# Hyperparameters:
# lr: 0.0012445908435230314
# epsilon: 8.506993746480856e-06
# beta_1: 0.9177777879149707
# Score: 0.00533091789111495

# Trial 02 summary
# Hyperparameters:
# lr: 0.001281303241718214
# epsilon: 1.1313488875509762e-06
# beta_1: 0.8295431321293527
# Score: 0.005345640704035759

# Trial 22 summary
# Hyperparameters:
# lr: 0.001861401947777535
# epsilon: 8.685222678115228e-06
# beta_1: 0.8057865255181482
# Score: 0.005346257705241442



# Showing 10 best trials Experiment 3 ------------------------------------------------------
# Trial 30 Complete [00h 52m 09s]
# val_loss: 0.004676087759435177
# Best val_loss So Far: 0.004265192896127701
# Total elapsed time: 1d 00h 54m 59s
#Search space summary
lr = {'default': 0.005, 'conditions': [], 'min_value': 0.005, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
epsilon = {'default': 1e-07, 'conditions': [], 'min_value': 1e-07, 'max_value': 1e-06, 'step': None, 'sampling': 'log'}
beta_1 = {'default': 0.7, 'conditions': [], 'min_value': 0.7, 'max_value': 0.95, 'step': None, 'sampling': 'linear'}
# Trial 15 summary
# Hyperparameters:
# lr: 0.006923375185711218
# epsilon: 1.984571030380454e-07
# beta_1: 0.8442654629126807
# Score: 0.004265192896127701

# Trial 29 summary
# Hyperparameters:
# lr: 0.006345862044688259
# epsilon: 1.2749613877742427e-07
# beta_1: 0.7458588760464884
# Score: 0.004676087759435177

# Trial 16 summary
# Hyperparameters:
# lr: 0.008916370246589792
# epsilon: 3.8274464228869826e-07
# beta_1: 0.8538515782936682
# Score: 0.004708214197307825

# Trial 01 summary
# Hyperparameters:
# lr: 0.006533120833007976
# epsilon: 4.4802643329892343e-07
# beta_1: 0.7597881983651766
# Score: 0.004753266926854849

# Trial 26 summary
# Hyperparameters:
# lr: 0.006480448609545671
# epsilon: 7.435740607050862e-07
# beta_1: 0.9355025971742776
# Score: 0.004882466979324818

# Trial 05 summary
# Hyperparameters:
# lr: 0.006239367203777412
# epsilon: 3.236388963604789e-07
# beta_1: 0.7354974407756975
# Score: 0.005121518857777119

# Trial 10 summary
# Hyperparameters:
# lr: 0.006114614547299648
# epsilon: 3.1273367531538505e-07
# beta_1: 0.9102152796920815
# Score: 0.005148760974407196

# Trial 22 summary
# Hyperparameters:
# lr: 0.005291237753935805
# epsilon: 5.88211156998908e-07
# beta_1: 0.899181331367268
# Score: 0.005149449221789837

# Trial 03 summary
# Hyperparameters:
# lr: 0.00999284955468417
# epsilon: 6.84624553643233e-07
# beta_1: 0.7188508120989483
# Score: 0.005237020086497068

# Trial 21 summary
# Hyperparameters:
# lr: 0.009213132647106437
# epsilon: 8.699961988209932e-07
# beta_1: 0.8325879178950081
# Score: 0.005305700935423374



# Showing 10 best trials Experiment 4 ------------------------------------------------------
# Trial 30 Complete [00h 51m 47s]
# val_loss: 0.005505845881998539
# Best val_loss So Far: 0.00467543862760067
# Total elapsed time: 1d 00h 53m 03s
# Search space summary
lr = {'default': 0.005, 'conditions': [], 'min_value': 0.005, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
epsilon = {'default': 1e-06, 'conditions': [], 'min_value': 1e-06, 'max_value': 1e-05, 'step': None, 'sampling': 'log'}
beta_1 = {'default': 0.7, 'conditions': [], 'min_value': 0.7, 'max_value': 0.95, 'step': None, 'sampling': 'linear'}
# Trial 05 summary
# Hyperparameters:
# lr: 0.009578077231644088
# epsilon: 2.801290554825321e-06
# beta_1: 0.8657329882043239
# Score: 0.00467543862760067

# Trial 00 summary
# Hyperparameters:
# lr: 0.00959883917289962
# epsilon: 3.3296355583287185e-06
# beta_1: 0.883027628878419
# Score: 0.0047065215185284615

# Trial 13 summary
# Hyperparameters:
# lr: 0.007039662952024265
# epsilon: 3.4010608652574186e-06
# beta_1: 0.7860603176187391
# Score: 0.004944487940520048

# Trial 12 summary
# Hyperparameters:
# lr: 0.005062812669939188
# epsilon: 5.640252917281334e-06
# beta_1: 0.8006590352548375
# Score: 0.004971934948116541

# Trial 23 summary
# Hyperparameters:
# lr: 0.005925384359950263
# epsilon: 1.0373203528599616e-06
# beta_1: 0.8805853106814601
# Score: 0.00504247797653079

# Trial 01 summary
# Hyperparameters:
# lr: 0.008669478283726398
# epsilon: 9.951809875014272e-06
# beta_1: 0.7781200768031261
# Score: 0.005140376277267933

# Trial 08 summary
# Hyperparameters:
# lr: 0.005585291941693512
# epsilon: 2.9153144375543712e-06
# beta_1: 0.8634692444262059
# Score: 0.00533894170075655

# Trial 15 summary
# Hyperparameters:
# lr: 0.00680825095507282
# epsilon: 1.1294836042017014e-06
# beta_1: 0.8405847262399777
# Score: 0.005408301018178463

# Trial 29 summary
# Hyperparameters:
# lr: 0.009303394878977092
# epsilon: 8.100329127584746e-06
# beta_1: 0.7765349415226285
# Score: 0.005505845881998539

# Trial 24 summary
# Hyperparameters:
# lr: 0.008933994170517153
# epsilon: 1.3236325964295175e-06
# beta_1: 0.8350952214169249
# Score: 0.005509244743734598