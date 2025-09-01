import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from params import parser
from dataset.circular_arc import CircularArcDataset
from model import DiRNN

args = parser.parse_args()
device = torch.device('cuda')
dataloader = DataLoader(CircularArcDataset(num_samples=64000, seq_len=64), batch_size=32, shuffle=True)
model = DiRNN().cuda()
optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

if __name__ == '__main__':
    info = {'train_loss': '#.###'}
    past = lambda x: torch.cat([torch.randn_like(x[:, :1]), x[:, :-1]], dim=1)
    for epoch in range(args.epochs):
        p_bar = tqdm(total=len(dataloader), postfix=info, )
        model.train()
        for i, x in enumerate(dataloader):
            x = x.cuda()
            e = torch.randn_like(x)
            out, _ = model(e, past_x=past(x))
            loss = F.mse_loss(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.update(1)
            info['train_loss'] = '%.3f' % loss.item()
            p_bar.set_postfix(info, refresh=True)
        p_bar.close()

        num_eval = 8
        model.eval()
        with torch.no_grad():
            past_h = torch.zeros(2, num_eval, 256).to(device)
            past_x = torch.randn(num_eval, 1, 2).to(device)
            x_seq = []
            for i in range(256):
                e = torch.randn(num_eval, 1, 2).to(device)
                x, h = model(e, h=past_h, past_x=past_x)
                x_seq.append(x)
                past_x = x
                past_h = h
            x_seq = torch.cat(x_seq, dim=1)
            plt.figure(figsize=(5, 5))
            x_seq = x_seq.cpu().numpy()
            for i in range(num_eval):
                plt.plot(x_seq[i, :, 0], x_seq[i, :, 1])
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            # plt.show()
            plt.savefig(f'results/deterministic_{epoch}.png')

        scheduler.step()