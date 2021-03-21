# This module loads and prepares the data

import torch, time, sys, re
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

ALPHABET = 'ACDEFGHIKLMNPQRSTVWXYZ-'
SEQ2IDX = dict(map(reversed, enumerate(ALPHABET)))

data_path = '/data/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m'
labels_path = '/data/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_LABELS.a2m'
mutations_path = '/data/BLAT_ECOLX_Ranganathan2015.csv'


def fasta(file_path):
    """This function parses a subset of the FASTA format
    https://en.wikipedia.org/wiki/FASTA_format"""

    print(f"Parsing fasta '{file_path}'")
    data = {
        'ur_up_': [], 'accession': [],
        'entry_name': [], 'offset': [],
        'taxonomy': [], 'sequence': []
    }

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()

            if line[0] == '>':
                key = line[1:]

                if i == 0:
                    name, offset = key.split("/")
                    ur_up_, acc = None, None
                else:
                    ur_up_, acc, name_offset = key.split("|")
                    name, offset = name_offset.split('/')

                data['ur_up_'].append(ur_up_)
                data['accession'].append(acc)
                data['entry_name'].append(name)
                data['offset'].append(offset)
                data['sequence'].append('')
                data['taxonomy'].append(name.split('_')[1])
            else:
                data['sequence'][-1] += line

            if i and (i % 50000 == 0):
                print(f"Reached: {i}")

    return pd.DataFrame(data=data)


def labels(labels_file, labels=[]):
    """Parses the labels file"""

    print(f"Parsing labels '{labels_file}'")
    with open(labels_file, 'r') as f:
        for i, line in enumerate(f):
            labels.append(line.split(':')[-1].strip())
    return pd.Series(labels)


def trim(full_sequences, focus_columns, sequences=[]):
    """Trims the sequences according to the focus columns"""

    for seq in full_sequences:
        seq = seq.replace('.', '-')
        trimmed = [seq[idx].upper() for idx in focus_columns]
        sequences.append(''.join(trimmed))
    return pd.Series(sequences)


def encode(sequences):
    t0 = time.time()
    print(f"Generating {len(sequences)} 1-hot encodings")
    tensors, l = [], len(ALPHABET)
    for seq in sequences:
        idxseq = [SEQ2IDX[s] for s in seq]
        tensor = F.one_hot(torch.tensor(idxseq), l).t().float()
        tensors.append(tensor)
    r = torch.stack(tensors)
    print(f"Generating {len(sequences)} 1-hot encodings. Took {round(time.time() - t0, 3)}s", r.shape)
    return r


def mutants(df):
    global mdf, offset, wt_full

    col = '2500'  # name of the column of our interest.
    mdf = pd.read_csv('data/BLAT_ECOLX_Ranganathan2015.csv')
    mdf = pd.DataFrame(data={'value': mdf[col].values}, index=mdf['mutant'].values)
    wt_row = df.iloc[0]  # wildtype row in df
    wt_off = wt_row['offset']  # wildtype offset (24-286)
    offset = int(wt_off.split('-')[0])  # left-side offset: 24
    wt_full = wt_row['sequence']
    focus_columns = [idx for idx, char in enumerate(wt_full) if char.isupper()]

    reg_co = re.compile("([a-zA-Z]+)([0-9]+)([a-zA-Z]+)")
    mutants = {'mutation': [], 'sequence': [], 'value': []}

    for i, (k, v) in enumerate(mdf.iterrows()):
        v = v['value']
        _from, _index, _to = reg_co.match(k).groups()
        _index = int(_index) - offset

        if wt_full[_index].islower():
            continue  # we skip the lowercase residues

        if wt_full[_index] != _from:
            print("WARNING: Mutation sequence mismatch:", k, "full wt index:", _index)

        mutant = wt_full[:_index] + _to + wt_full[_index + 1:]
        mutant_trimmed = [mutant[idx] for idx in focus_columns]

        mutants['mutation'].append(k)
        mutants['sequence'].append(''.join(mutant_trimmed))
        mutants['value'].append(v)
    return pd.DataFrame(data=mutants)


def hamming_distance(a, b):
    result = 0
    for x, (i, j) in enumerate(zip(a, b)):
        if i != j:
            # print(f'char not math{i, j}in {x}')
            result += 1
    return result


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v

    return v / norm


def seq_weights(df):
    theta = 0.2  # 0.01 for viral proteins?
    weights = []

    for i in range(df.shape[0]):
        hamming_dist = []
        for j in range(df.shape[0]):
            hamming_dist.append(hamming_distance(df['trimmed'][i], df['trimmed'][j]))

        norm_dist = normalize(hamming_dist) #[float(dist) / sum(hamming_dist) for dist in hamming_dist]

        weights.append(1 / sum([1 for norm in norm_dist if norm < theta]))

    n_eff = sum(weights)
    p_s = [w / n_eff for w in weights]

    return p_s


def data(batch_size=128, device='cpu'):
    df = fasta(data_path)
    df['label'] = labels(labels_path)

    # First sequence in the dataframe/fasta file is our wildtype.
    wildtype_seq = df.sequence[0]

    # What wildtype column-positions are we confident about (uppercased chars)
    focus_columns = [idx for idx, char in enumerate(wildtype_seq) if char.isupper()]

    # Trim the full sequences according to the columns we are confident at
    df['trimmed'] = trim(df.sequence, focus_columns)

    # Unique aminoacids are are:
    # ''.join(set(''.join(df.trimmed.to_list())))

    dataset = encode(df.trimmed).to(device)

    weights = seq_weights(df)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)  # shuffle=True,

    mutants_df = mutants(df)
    mutants_tensor = encode(mutants_df.sequence)

    return dataloader, df, mutants_tensor, mutants_df


# nice colors for the terminal
class c:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    dataloader, df, mutants_tensor, mutants_df = data()
