from core.astromer import get_ASTROMER, train, predict
from core.data  import pretraining_records
from sklearn.metrics import r2_score, mean_squared_error
import os, sys, json

dataset_path = './data/records/macho/'
exp_path = 'weights/astromer_10022021'
conf_file = os.path.join(exp_path, 'conf.json')
with open(conf_file, 'r') as handle:
    conf = json.load(handle)

test_batches = pretraining_records(os.path.join(dataset_path, 'test'),
                                    256,
                                    max_obs=conf['max_obs'],
                                    msk_frac=conf['msk_frac'],
                                    rnd_frac=conf['rnd_frac'],
                                    same_frac=conf['same_frac'],
                                    sampling=False, shuffle=False)

astromer = get_ASTROMER(num_layers=conf['layers'],
                        d_model=conf['head_dim'],
                        num_heads=conf['heads'],
                        dff=conf['dff'],
                        base=conf['base'],
                        use_leak=conf['use_leak'],
                        dropout=conf['dropout'],
                        maxlen=conf['max_obs'])

weights_path = 'weights/astromer_10022021/weights'
astromer.load_weights(weights_path)

result = predict(astromer, test_batches.take(10), conf)

print('r2: {}\nmse: {}'.format(r2_score(result['x_true'][...,0], result['x_pred'][...,0]),
                               result['mse']))
