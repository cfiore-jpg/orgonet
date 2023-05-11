import pickle

names_file = 'data/train/trainnames.txt'
elements_file = 'data/train/trainelements.txt'
positions_file = 'data/train/trainpositions.txt'
sizes_file = 'data/train/trainsizes.txt'

names_pkl = 'data/names.pkl'
elements_pkl = 'data/elements.pkl'
positions_pkl = 'data/positions.pkl'
sizes_pkl = 'data/sizes.pkl'

names, elements, positions, sizes = [], [], [], []
try:
    with open(names_pkl, 'rb') as f:
        names = pickle.load(f)

    with open(elements_pkl, 'rb') as f:
        elements = pickle.load(f)
    
    with open(positions_pkl, 'rb') as f:
        positions = pickle.load(f)

    with open(sizes_pkl, 'rb') as f:
        sizes = pickle.load(f)

except:
    with open(names_file, 'r') as f:
        names = f.readlines()

    with open(elements_file, 'r') as f:
        elements = f.readlines()

    with open(positions_file, 'r') as f:
        positions = f.readlines()

    with open(sizes_file, 'r') as f:
        sizes = f.readlines()

    m = {}
    for n, e, p, s in zip(names, elements, positions, sizes):
        m[n.strip()] = (e.strip(), p.strip(), s.strip())

    names = list(m.keys())
    elements, positions, sizes = [], [], []
    for i, (e, p, s) in enumerate(m.values()):
        elements.append(e[:-1].split(','))
        positions.append([[float(p__) for p__ in p_[1:-1].split()] for p_ in p[:-1].split(',')])
        sizes.append(int(s))
        print('Parsing... {}/{}'.format(i, len(m.values())), end='\r')
    
    with open(names_pkl, 'wb') as f:
        pickle.dump(names, f)

    with open(elements_pkl, 'wb') as f:
        pickle.dump(elements, f)
    
    with open(positions_pkl, 'wb') as f:
        pickle.dump(positions, f)

    with open(sizes_pkl, 'wb') as f:
        pickle.dump(sizes, f)


