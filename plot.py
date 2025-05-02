
losses = []
f = open('/mnt/disk90/user/gkaynar/mrnadesign/policy_network_and_mfe_prediction/structure_last4.out', 'r')
for lines in f:
    if lines.startswith('Epoch'):
        loss = lines.split(':')[-1].strip()
        losses.append(float(loss))
    
import matplotlib.pyplot as plt
x  = range(len(losses))
plt.plot(x, losses,label='train_loss')
plt.ylabel('Training loss')
plt.legend()
plt.xlabel('Epochs')
plt.savefig('training.png')

#print(losses)

f.close()