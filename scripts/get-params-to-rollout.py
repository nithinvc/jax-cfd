import argparse
import json


def sbatch_helper(json_fp):
    data = json.load(open(json_fp, 'r'))
    processed_paths = [x['rollout_path'] for x in data]
    viscosities = [x['averaged_param'] for x in data]
    seed = int(json_fp.split('_')[-2])

    processed_paths = [
        p.replace(f'/global/cfs/cdirs/m4558/shared/meta-pde/evals/seed_{seed}/', '') for p in processed_paths
    ]
    processed_paths = [p[:-1] if p.endswith('/') else p for p in processed_paths]

    paths = ""
    for p in processed_paths:
        paths += f" \"{p}\""
    print(paths)
    vis = ""
    for v in viscosities:
        vis += f" \"{v}\""
    print(vis)
    print('total number of rollouts: ', len(viscosities))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--ic_index", type=int, default=0)
    parser.add_argument("--processed", action='store_true', help="whether the json file is already processed")
    args = parser.parse_args()

    if args.processed:
        sbatch_helper(args.json)
        exit()

    data = json.load(open(args.json, 'r'))
    ic_index = args.ic_index
    seed = args.json.split('.')[0].split('_')[-1]

    predicted_params = [x['predicted_param'] for x in data]
    import numpy as np
    predicted_params = [x[ic_index] for x in predicted_params]
    predicted_params = np.asarray(predicted_params)
    predicted_params = predicted_params.mean(axis=1)


    # modify the json file
    new_fp = args.json.replace('.json', f'_ic_{ic_index}_seed_{seed}_processed.json')
    from copy import deepcopy
    new_objs = deepcopy(data)
    for i, obj in enumerate(new_objs):
        obj['averaged_param'] = predicted_params[i].item()
        obj['rollout_path'] = f'/global/cfs/cdirs/m4558/shared/meta-pde/evals/seed_{seed}/ic_{ic_index}_param_{predicted_params[i].item()}/'

    json.dump(new_objs, open(new_fp, 'w'), indent=4)
    

if __name__ == "__main__":
    main()