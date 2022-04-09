import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
	parser.add_argument('--batch', default=258, type=int, help='batch size')
	parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--dim', default=64, type=int, help='embedding size')
	parser.add_argument('--data', type=str, default='beibei',help='name of dataset')
	
	parser.add_argument('--weighted',type=bool,default=True, help='used the weighted adj matrix')
	parser.add_argument('--ls',type=float,default=0.5, help='decay factor of contrastive learning loss')
	parser.add_argument("--n_hops", type=int, default=2, help="shufflet the data")
	parser.add_argument("--shuffle", type=bool, default=True, help="shufflet the data")
	parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
	parser.add_argument("--gpu_id", type=int, default=3, help="gpu id")
	parser.add_argument('--Ks', nargs='?', default='[1, 3, 5, 7, 10]', help='top-K list')
	parser.add_argument("--batch_test_flag", type=bool, default=True, help="batch items")
	parser.add_argument('--test_flag', nargs='?', default='part',
						help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
	parser.add_argument("--save", type=bool, default=True, help="save model or not")
	parser.add_argument("--out_dir", type=str, default="./save/", help="output directory for model")
	
	return parser.parse_args()
args = parse_args()
