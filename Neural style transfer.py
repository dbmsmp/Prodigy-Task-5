import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

da = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ak_img(p, m=400):
 a = Image.open(p).convert('RGB')
 s = max(a.size) if max(a.size) <= m else m
 t = transforms.Compose([transforms.Resize(s), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
 return t(a)[:3,:,:].unsqueeze(0).to(da)

def dalela(x):
 x = x.to("cpu").clone().detach().squeeze(0)
 x = x.numpy().transpose(1,2,0)
 x = x*(0.229,0.224,0.225)+(0.485,0.456,0.406)
 return Image.fromarray((x.clip(0,1)*255).astype('uint8'))

class Aakarshi(nn.Module):
 def __init__(s,c,s_):
  super().__init__()
  s.vgg = models.vgg19(pretrained=True).features.to(da).eval()
  s.c = c
  s.s = s_
  s.w = {'0':1e3,'5':1e2,'10':1e2,'19':1e2,'28':1e2}
  s.g = lambda x: torch.einsum("bchw,bdhw->bcd",x,x)/(x.shape[1]*x.shape[2]*x.shape[3])
 def forward(s,x):
  f = {}
  for n,l in s.vgg._modules.items():
   x = l(x)
   if n in s.c: f['c'] = x
   if n in s.s: f[f's{n}'] = s.g(x)
  return f

def run(a,b,o):
 c = ak_img(a)
 s = ak_img(b, max_size=c.shape[-1])
 x = c.clone().requires_grad_(True)
 m = Aakarshi(['21'], ['0','5','10','19','28'])
 opt = optim.Adam([x], lr=0.003)
 for _ in range(501):
  tf, cf, sf = m(x), m(c), m(s)
  cl = torch.mean((tf['c'] - cf['c']) ** 2)
  sl = sum(m.w[n]*torch.mean((tf[f's{n}'] - sf[f's{n}'])**2) for n in m.s)
  loss = cl + sl
  opt.zero_grad()
  loss.backward()
  opt.step()
 dalela(x).save(o)

run("content.jpg", "style.jpg", "aakarshi_dalela.jpg")



