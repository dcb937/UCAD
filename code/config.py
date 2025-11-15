from dataloaders.dataset import ACDCDataSets, SynapseDataSets

def parse_args(args):
    if 'ACDC' == args.exp.split("/")[0]:
        args.root_path = '/data2/chengboding/data/ACDC'
        args.train_list = 'train_slices.list'
        args.test_list = 'val.list'
        args.num_classes = 4
        args.labelnum = 7  # 10%: 7 #5%: 3  1%: 1
        args.labeled_num = args.labelnum
        args.patch_size = [256, 256]
        args.test_interval = 200

        CustomDataset = ACDCDataSets
    
    elif 'Synapse' == args.exp.split("/")[0]:
        args.root_path = '/data/chengboding/data/synapse'
        args.train_list = 'train.txt'
        args.test_list = 'test_vol.txt'
        args.num_classes = 9
        args.labelnum = 1  # 10%: 2  # 5%: 1
        args.labeled_num = args.labelnum
        args.patch_size = [256, 256]
        args.test_interval = 300
        CustomDataset = SynapseDataSets

    else:
        raise NotImplementedError(f'Not Implemented Error "{args.exp}"')
    
    return CustomDataset
    

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Synapse" in dataset:
        ref_dict = {"1": 93, "2": 256, "4": 522, "18": 2211,"20": 2497}
    else:
        raise NotImplementedError(f'Not Implemented Error "{dataset}"')
    return ref_dict[str(patiens_num)]
