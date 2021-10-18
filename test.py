import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from utils import img2tensor, tensor2img
from model import MultiLevelAE_OST


def main():
    parser = argparse.ArgumentParser(description='Optimal Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--result_path', '-r', type=str, default='results/',
                        help='Folder to save results into')
    parser.add_argument('--alpha', '-a', type=float, default=1.0,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(negative value indicate CPU)')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = MultiLevelAE_OST()
    model = model.to(device)
    
    os.makedirs(args.result_path, exist_ok=True)
    
    c = Image.open(args.content).convert('RGB')
    s = Image.open(args.style).convert('RGB')
    c_tensor = img2tensor(c).to(device)
    s_tensor = img2tensor(s).to(device)
    
    with torch.no_grad():
        out = model(c_tensor, s_tensor, args.alpha)
    
    c_name = os.path.splitext(os.path.basename(args.content))[0]
    s_name = os.path.splitext(os.path.basename(args.style))[0]
    output_name = f'{c_name}_{s_name}'
    
    out = tensor2img(out)
    out.save(os.path.join(args.result_path, f'{output_name}.jpg'))

    demo = Image.new('RGB', (c.width * 2, c.height))
    out = out.resize(c.size)
    s = s.resize((i // 4 for i in c.size))

    demo.paste(c, (0, 0))
    demo.paste(out, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    demo.save(os.path.join(args.result_path, f'{output_name}_style_transfer_demo.jpg'), quality=95)

    out.paste(s,  (0, out.height - s.height))
    out.save(os.path.join(args.result_path, f'{output_name}_with_style_image.jpg'), quality=95)

    print(f'results saved into "{args.result_path}" folder, files starting with "{output_name}"')
    

if __name__ == '__main__':
    main()
