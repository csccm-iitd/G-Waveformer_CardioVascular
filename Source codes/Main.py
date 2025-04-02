#Loading packages
from utilities3 import *
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from timeit import default_timer
from spectral_convolution import FactorizedSpectralConv
from integral_transform import IntegralTransform
from neighbor_search import NeighborSearch
# from tfnogno import FNOGNO  
from wwnogno import WNOGNO
# device= torch.device('cuda:1')
#%%
%load the data
y_trainv = torch.tensor(np.load('datasets/vel_4.npy'),dtype = torch.float)
y_trainp = torch.tensor(np.load('datasets/press_4.npy'),dtype = torch.float)
ntrain = 32
ntest = 5

y_train1d = y_trainv[0:ntrain,0:20:2,:].permute(0,2,1)
y_train2d  = y_trainv[0:ntrain,20:60:2,:].permute(0,2,1)

y_normalizer = UnitGaussianNormalizer(y_train1d[:,:,0:1].unsqueeze(1))

y_train1 = torch.zeros(y_train1d.shape)
for i in range(y_train1d.shape[-1]):
    y_train1[:,:,i:i+1] = y_normalizer.encode(y_train1d[:,:,i:i+1].unsqueeze(1)).squeeze(1)
    
y_train2 = torch.zeros(y_train2d.shape)
for i in range(y_train2d.shape[-1]):
    y_train2[:,:,i:i+1] = y_normalizer.encode(y_train2d[:,:,i:i+1].unsqueeze(1)).squeeze(1)

# y_train1 = y_normalizer.encode(y_train1)
# y_train2 = y_normalizer.encode(y_train2)

y_test1d = y_trainv[ntrain:ntrain+ntest,0:20:2,:].permute(0,2,1)
y_test1 = torch.zeros(y_test1d.shape)
for i in range(y_test1d.shape[-1]):
    y_test1[:,:,i:i+1] = y_normalizer.encode(y_test1d[:,:,i:i+1].unsqueeze(1)).squeeze(1)
    
y_test2 = y_trainv[ntrain:ntrain+ntest,20:80:2,:].permute(0,2,1)

# y_test1 = y_normalizer.encode(y_test1)
# y_test2 = y_normalizer.encode(y_test2)

x_cord_dash =  torch.tensor(np.load('datasets/cord_4.npy'),dtype =torch.float)[0]
x_cord_d1 = 2*(x_cord_dash[:,0]-torch.min(x_cord_dash[:,0]))/(torch.max(x_cord_dash[:,0])-torch.min(x_cord_dash[:,0]))-1
x_cord_d2 = 2*(x_cord_dash[:,1]-torch.min(x_cord_dash[:,1]))/(torch.max(x_cord_dash[:,1])-torch.min(x_cord_dash[:,1]))-1
x_cord_d3 = 2*(x_cord_dash[:,2]-torch.min(x_cord_dash[:,2]))/(torch.max(x_cord_dash[:,2])-torch.min(x_cord_dash[:,2]))-1
x_cord = torch.cat([x_cord_d1[:,None],x_cord_d2[:,None],x_cord_d3[:,None]],dim=1)

epochs = 120
learning_rate = 0.0008
batch_size = 1
step_size = 5
gamma = 0.8
T_out = 40
T_in = 10
step = 1
 
train_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train1, y_train2), batch_size=batch_size, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test1, y_test2), batch_size=1, shuffle=False)

em_d = 23
# xt1 = np.linspace(torch.min(x_cord[:,0]).numpy(),torch.max(x_cord[:,0]).numpy(), em_d)
# yt1 = np.linspace(torch.min(x_cord[:,1]).numpy(),torch.max(x_cord[:,1]).numpy(), em_d)
# zt1 = np.linspace(torch.min(x_cord[:,2]).numpy(),torch.max(x_cord[:,2]).numpy(), em_d)

xt1 = np.linspace(-1,1, em_d)
yt1 = np.linspace(-1,1, em_d)
zt1 = np.linspace(-1,1, em_d)

Xt1, Yt1 , Zt1 = np.meshgrid(xt1, yt1,zt1)
X_f_train = np.hstack([Xt1.reshape(em_d*em_d*em_d, 1), Yt1.reshape(em_d*em_d*em_d, 1),Zt1.reshape(em_d*em_d*em_d, 1)])
x_cord1 = torch.tensor(X_f_train, dtype=torch.float)
#%%
#Model
Search_neib = NeighborSearch(use_open3d=False)
radius = torch.tensor([0.11],dtype=torch.float)
nbr = Search_neib(x_cord,x_cord1,radius)

radius2 = torch.tensor([0.11],dtype=torch.float)
nbr2 = Search_neib(x_cord1,x_cord,radius2)

out_channels = 1
in_channels = 11

%Model specifications
torch.manual_seed(0)
np.random.seed(0)


from wwnogno import WNOGNO
model = WNOGNO(in_channels=16,
            out_channels=10,
            in_channels2=9,
            in_channels3=7,
            out_channels2=1,
            projection_channels = 64,
            embed_dim = 23,
            wavelet = 'db6',
            level = 2,        
            width = 9,       
            layers = 2,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_mlp_hidden_layers=[64,32],
            gno_mlp_non_linearity=F.mish, 
            gno_transform_type=2,
            gno_use_open3d=False)
#%%
""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_dl = 0
    for xx, yy in train_loader1:  
        optimizer.zero_grad()
        data_loss = 0
        # x = x.to(device)
        # y = y.to(device) 
        # for k in range(0,xx.shape[0]):
        x = xx.squeeze(0)
        y = yy.squeeze(0)
        for t in range(0, T_out-T_in, step):
            z = y[:,t:t+step]
            im =  model(x,x_cord,x_cord1,nbr,nbr2)  
            imd = y_normalizer.decode(im.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            x = torch.cat((x[:, 1:], im), dim=-1)
            # y = torch.cat((x[..., 1:], im), dim=-1)
            data_loss += F.mse_loss(im, z) 

             
        loss = data_loss
        train_l2 =  loss.item()
        train_dl =  data_loss.item()
        
        loss.backward()
        optimizer.step()
        
        
    scheduler.step()
    model.eval()
    
    test_l2_step = 0
    test_l2_full = 0

    with torch.no_grad():
        for xx, yy in test_loader2:
            loss = 0
            x = xx.squeeze(0)
            y = yy.squeeze(0)
            for t in range(0, T_out-T_in, step):
                z = y[:,t:t+step]
                im =  model(x,x_cord,x_cord1,nbr,nbr2) 
                imd= y_normalizer.decode(im.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                loss += torch.norm(imd-z, p=2)/torch.norm(z, p=2)
                if t == 0:
                   pred = imd
                else:
                   pred = torch.cat((pred, imd), -1)
                x = torch.cat((x[..., 1:], im), dim=-1)
                # y = torch.cat((x[..., 1:], im), dim=-1)
    
                
            test_l2_step += loss.item()
            test_l2_full += (torch.norm(pred-y, p=2)/torch.norm(y, p=2)).item()
                    
            
    train_l2 /= (ntrain*T_out)
    test_l2_step /= (ntest*T_out)
    test_l2_full /= (ntest*T_out)
    t2 = default_timer()
    if ep%5==1:
        torch.save(model, 'model/model_geowto_11')
    print('Epoch %d - Time %0.4f - Train %0.6f - test Data %0.6f - Test %0.6f' 
          % (ep, t2-t1, train_l2, test_l2_step, test_l2_full)) 
#%%
model = torch.load('model/model_geowto_11')
# %%
y_test1d = y_trainv[10:20,0:20:2,:].permute(0,2,1)
y_test1 = torch.zeros(y_test1d.shape)
for i in range(y_test1d.shape[-1]):
    y_test1[:,:,i:i+1] = y_normalizer.encode(y_test1d[:,:,i:i+1].unsqueeze(1)).squeeze(1)
    
y_test2 = y_trainv[10:20,20:60:2,:].permute(0,2,1)

test_loader3 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test1, y_test2), batch_size=1, shuffle=False)
#%%
""" Prediction """
pred0 = torch.zeros(y_test2.shape)
index = 0
test_e = torch.zeros(y_test2.shape)        
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
t1 = default_timer()
with torch.no_grad():
     for xx, yy in test_loader3:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        x = xx.squeeze(0)
        y = yy.squeeze(0)
        for t in range(0, T_out-T_in, step):
            z = y[:,t:t+step]
            im =  model(x,x_cord,x_cord1,nbr,nbr2) 
            imd = y_normalizer.decode(im.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            loss += torch.norm(imd-z, p=2)/torch.norm(z, p=2)
            if t == 0:
               pred = imd
            else:
               pred = torch.cat((pred, imd), -1)
            x = torch.cat((x[..., 1:], im), dim=-1)
  
        pred0[index,:,:] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-y, p=2)/torch.norm(y, p=2)).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntest/ (T_out/step), test_l2_full/ ntest)
        index = index + 1
t2 = default_timer()
print(t2-t1)       
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T_out/step), '%')
# torch.save(model, 'model/model_geowto')
#%%
pred_s = pred0[9,1000:3000,0:20].reshape(-1)/1000
test_s = y_test2[9,1000:3000,0:20].reshape(-1)/1000
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plt.scatter(pred_s,test_s)
# plt.scatter(test_s,test_s)
figure1 = plt.figure(figsize = (18, 8))
""" Plotting """ 
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.size'] = 16
plt.subplot(1,2, 1)
plt.scatter(test_s,pred_s,linewidth=1,color= 'r',label='R$^2$ value = 0.8990'); 
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='b', lw=2, scalex=False, scaley=False)
# plt.plot(loss_nagumo[0:400], label='Nagumo', linewidth=2,color= 'g'); 
# plt.plot(loss_allencan[0:400], label='Allencan', linewidth=2,color= 'k'); 
# plt.plot(loss_burger, label='1D Burgers', linewidth=2); plt.plot(loss_burger[0], color= 'r')
# plt.plot(loss_poissons, label='Poissons', linewidth=2); plt.plot(loss_poissons[0],color= 'b')
# plt.plot(loss_nagumo, label='Nagumo', linewidth=2); plt.plot(loss_nagumo[0],color= 'g')
plt.xlabel('Ground truth flowrate', fontweight='bold')
plt.ylabel('Predicted flowrate', fontweight='bold')
plt.legend()
plt.xlim([100,175])
plt.ylim([100,175])
plt.grid(True)
figure1.savefig('scatter_data4_v.pdf', format='pdf', dpi=300, bbox_inches='tight')
figure1.savefig('scatter_data4_v.png', format='png', dpi=300, bbox_inches='tight')
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
res = stats.linregress(test_s,pred_s)
r2 = res.rvalue**2
print(r2)
#%%
""" Prediction """
pred0 = torch.zeros(y_test2.shape)
index = 0
test_e = torch.zeros(y_test2.shape)        
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
t1 = default_timer()
with torch.no_grad():
     for xx, yy in test_loader2:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        x = xx.squeeze(0)
        y = yy.squeeze(0)
        for t in range(0, T_out-T_in, step):
            z = y[:,t:t+step]
            im =  model(x,x_cord,x_cord1,nbr,nbr2) 
            imd = y_normalizer.decode(im.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            loss += torch.mean(imd-z)**2/torch.mean(z)**2
            if t == 0:
               pred = imd
            else:
               pred = torch.cat((pred, imd), -1)
            x = torch.cat((x[..., 1:], im), dim=-1)
  
        pred0[index,:,:] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-y)**2/torch.norm(y)**2).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntrain/ (T_out/step), test_l2_full/ ntrain)
        index = index + 1
t2 = default_timer()
print(t2-t1)       
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T_out/step), '%')
#%%
x_cord = x_cord.cpu().detach().numpy()
#%%
""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(figsize = (14,10))
fig.text(0.04,0.20,'\n Error', rotation=90, color='purple', fontsize= 15)
fig.text(0.04,0.48,'\n Truth', rotation=90, color='green', fontsize= 15)
fig.text(0.04,0.70,'\n Prediction', rotation=90, color='red', fontsize= 15)
plt.subplots_adjust(hspace=0.25,wspace=0.1)
s = 4
index = 0
valu1 = 1
valu2 = 11
valu3 = 20
valu4 = 24


# vm1 = torch.min(pred0)
# vm2 = torch.max(pred0)

cmapl = 'jet'
ax = fig.add_subplot(3,4,1, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= pred0[s,:,valu1:valu1+1].detach().numpy(), cmap= cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025)
# fig.colorbar(p)
plt.title(str(10+valu1+1),color='b', fontsize=15, fontweight='bold')
plt.margins(0)

# plt.subplot(3,4, index+1+4)
ax = fig.add_subplot(3,4,5, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= y_test2[s,:,valu1:valu1+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025)
# plt.title('Actual')
plt.margins(0)

# plt.subplot(3,4, index+1+8)
ax = fig.add_subplot(3,4,9, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= (100*(torch.absolute(pred0[s,:,valu1:valu1+1]-y_test2[s,:,valu1:valu1+1]))/y_test2[s,:,valu1:valu1+1]).detach().numpy(), cmap='jet', linewidth=0.4,vmin =0,vmax =100)
# plt.colorbar(p,fraction=0.025)
# plt.title('Error')
plt.margins(0)


ax = fig.add_subplot(3,4,2, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= pred0[s,:,valu2:valu2+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025)
# fig.colorbar(p)
plt.title(str(10+valu2+1),color='b', fontsize=15, fontweight='bold')
plt.margins(0)

# plt.subplot(3,4, index+1+4)
ax = fig.add_subplot(3,4, 6, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= y_test2[s,:,valu2:valu2+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025)
# plt.title('Actual')
plt.margins(0)

# plt.subplot(3,4, index+1+8)
ax = fig.add_subplot(3,4,10, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= (100*(torch.absolute(pred0[s,:,valu2:valu2+1]-y_test2[s,:,valu2:valu2+1]))/y_test2[s,:,valu2:valu2+1]).detach().numpy(), cmap='jet', linewidth=0.4,vmin =0,vmax =100)
# plt.colorbar(p,fraction=0.025)
# plt.title('Error')
plt.margins(0)



ax = fig.add_subplot(3,4,3, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= pred0[s,:,valu3:valu3+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025,  location ='right',pad = 0.15)
# fig.colorbar(p)
plt.title(str(10+valu3+1),color='b', fontsize=15, fontweight='bold')
plt.margins(0)

# plt.subplot(3,4, index+1+4)
ax = fig.add_subplot(3,4, 7, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= y_test2[s,:,valu3:valu3+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
# plt.colorbar(p,fraction=0.025,  location ='right',pad = 0.15)
# plt.title('Actual')
plt.margins(0)

# plt.subplot(3,4, index+1+8)
ax = fig.add_subplot(3,4,11, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= (100*(torch.absolute(pred0[s,:,valu3:valu3+1]-y_test2[s,:,valu3:valu3+1]))/y_test2[s,:,valu3:valu3+1]).detach().numpy(), cmap='jet', linewidth=0.4,vmin =0,vmax =100)
# plt.colorbar(p,fraction=0.025, location ='right',pad = 0.15)
# plt.title('Error')
plt.margins(0)



ax = fig.add_subplot(3,4,4, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= pred0[s,:,valu4:valu4+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
plt.colorbar(p,fraction=0.020,  location ='right',pad = 0.15)
# fig.colorbar(p)
plt.title(str(10+valu4+1),color='b', fontsize=15, fontweight='bold')
plt.margins(0)

# plt.subplot(3,4, index+1+4)
ax = fig.add_subplot(3,4, 8, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= y_test2[s,:,valu4:valu4+1].detach().numpy(), cmap=cmapl, linewidth=0.4,vmin =vm1,vmax = vm2)
plt.colorbar(p,fraction=0.020,  location ='right',pad = 0.15)
# plt.title('Actual')
plt.margins(0)

# plt.subplot(3,4, index+1+8)
ax = fig.add_subplot(3,4,12, projection='3d')
p = ax.scatter(x_cord[:,0:1],x_cord[:,1:2],x_cord[:,2:3], c= (100*(torch.absolute(pred0[s,:,valu4:valu4+1]-y_test2[s,:,valu4:valu4+1]))/y_test2[s,:,valu4:valu4+1]).detach().numpy(), cmap='jet', linewidth=0.4,vmin =0,vmax =100)
plt.colorbar(p,fraction=0.020, location ='right',pad = 0.15)
# plt.title('Error')
plt.margins(0)

