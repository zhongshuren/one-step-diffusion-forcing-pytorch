import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from params import parser
from dataset.circular_arc import CircularArcDataset
from model import DiRNN
from loss import DiffusionForcingLoss

args = parser.parse_args()
device = torch.device('cuda')
dataloader = DataLoader(CircularArcDataset(num_samples=64000, seq_len=64), batch_size=args.batch_size, shuffle=True)
model = DiRNN(dim=args.dim).to(device)
diffusion_loss = DiffusionForcingLoss(model)
optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

torch.manual_seed(args.seed)

if __name__ == '__main__':
    info = {'train_loss': '#.###'}
    for epoch in range(args.epochs):
        p_bar = tqdm(total=len(dataloader), postfix=info, )
        model.train()
        for i, x in enumerate(dataloader):
            x = x.to(device)
            loss = diffusion_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.update(1)
            info['train_loss'] = '%.3f' % loss.item()
            p_bar.set_postfix(info, refresh=True)
        p_bar.close()

        num_eval = args.num_eval
        eval_time_steps = 256
        model.eval()
        with torch.no_grad():
            past_h = torch.zeros(2, num_eval, args.dim).to(device) # GRU
            # past_h = (torch.zeros(2, num_eval, args.dim).to(device), torch.zeros(2, num_eval, args.dim).to(device))   # LSTM
            past_x = torch.zeros(num_eval, 1, 2).to(device)
            x_seq = []
            for i in range(eval_time_steps):
                if args.flow_steps == 1:
                    e = torch.randn(num_eval, 1, 2).to(device)
                    x, h = model(e, h=past_h, past_x=past_x)
                else:
                    eps = 1 / args.flow_steps
                    x = torch.randn(num_eval, 1, 2).to(device)
                    h = past_h
                    for i in range(args.flow_steps):
                        t1 = torch.tensor([i * eps]).to(device)
                        t2 = torch.tensor([(i + 1) * eps]).to(device)
                        x_, h = model(x, h=past_h, t1=t1, t2=t2, past_x=past_x)
                        x = x * (1 - eps) + x_ * eps
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
            plt.savefig(f'results/flow_forcing_{args.flow_steps}_{epoch}.png')

        scheduler.step()
