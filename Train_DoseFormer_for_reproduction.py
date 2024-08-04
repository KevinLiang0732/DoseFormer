from utils.utils import get_data_DynamicStatic
from Model_DoseFormer.DoseGuide import cnn_lstm_attention_gat
import datetime



path = 'data/data.pkl'
import pickle
def read_data(path = 'data/data.pkl',device='cpu'):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    for i in data:
        data[i] = data[i].to(device)
    return data['allseq2t'],data['allst'],data['idx_train'],data['idx_val'],data['idx_test'],data['y'],data['idx_train_dynamic'],data['idx_val_dynamic'],data['idx_test_dynamic'],data['idx_train_static'],data['idx_val_static'],data['idx_test_static']

allseq2t,allst,idx_train,idx_val,idx_test,y,idx_train_dynamic,idx_val_dynamic,idx_test_dynamic,idx_train_static,idx_val_static,idx_test_static = read_data(path,'cuda:1')


import torch.nn.functional as F
import torch
from utils.utils import accuracy, test_para

tg = cnn_lstm_attention_gat()
LR = 0.00005
tg.float()
tg.cuda(device=1)
loss_function = F.nll_loss
optimizer = torch.optim.Adam(tg.parameters(), lr=LR)

epochs = 2000000
import os

model_savepath = "Model_Save/_Train_DoseGuide_lxy/model_epo_{}_loss_{}_acc_{}.pkl"
train_log_path = "Model_Log/_Train_DoseGuide_lxy/train_log.txt"
val_log_path = "Model_Log/_Train_DoseGuide_lxy/val_log.txt"
test_log_path = "Model_Log/_Train_DoseGuide_lxy/test_log.txt"

name = '_Train_DoseGuide_lxy'
if not os.path.exists('Model_Log/{}'.format(name)):
    os.mkdir('Model_Log/{}'.format(name))
    print("create dir " + 'Model_Log/{}'.format(name))
if not os.path.exists('Model_Save/{}'.format(name)):
    os.mkdir('Model_Save/{}'.format(name))
    print("create dir " + 'Model_Save/{}'.format(name))

starttime = datetime.datetime.now()
for epoch in range(epochs):
    
    tg.train()
    out,_,_ ,adj= tg(allseq2t,allst)
    optimizer.zero_grad()
    loss = loss_function(out[idx_train],y[idx_train])
    loss.backward()
    optimizer.step()
    acc = accuracy(out[idx_train],y[idx_train])
    accp , lr = test_para(out,y,idx_train)

    sen = lr[4]
    spc = lr[5]
    losst = float(loss)
    acct = float(acc)

    if epoch % 100 ==0:
        endtime = datetime.datetime.now()
        print("epoch : {} , loss : {} , acc : {} , time : {}s".format(
            epoch,
            float(loss),
            float(acc),
            (endtime - starttime).seconds
        ))
        print(lr)
        starttime = datetime.datetime.now()
    if epoch % 100 ==0:
        with open(train_log_path,'a+') as f:
            print(
                epoch,
                float(loss),
                float(acc),
                lr,
                file=f
            )
    if epoch % 2000 ==0:
        if float(acc) >= 0.8:
            torch.save(tg, model_savepath.format(
                                                epoch,
                                                float(loss),
                                                float(acc)
                                                )
                                                )
                                    
        



    


    

    