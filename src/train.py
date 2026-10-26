import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from copy import deepcopy
import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .model import SVDKL, TrainOnlyTransform, full_checkpoint, load_full_checkpoint
from .predict import load_checkpoint, marker_catalog, predict_values, predict_full_values


HERE = Path(__file__).resolve().parent.parent


_BENCHMARK = {'models': [{'id': 'SVDKL-0.01M', 'kind': 'svdkl', 'width': 32, 'depth': 1, 'parameters': 9731},
            {'id': 'SVDKL-4.35M', 'kind': 'svdkl', 'width': 512, 'depth': 2, 'parameters': 4347395},
            {'id': 'SVDKL-46.46M', 'kind': 'svdkl', 'width': 1376, 'depth': 3, 'parameters': 46462019}],
 'svdkl': {'peak_lr': 0.005,
           'warmup': 10,
           'max_epochs': 105,
           'patience': 45,
           'schedule': 'linear',
           'heads': 4,
           'dropout': 0.1,
           'beta_init': 0.1,
           'delta_init': 1.0,
           'optimizer': 'AdamW',
           'weight_decay': 0.0,
           'grad_clip': 1.0,
           'top_k': 500},
 'accessions': ['SL001',
                'SL007',
                'SL008',
                'SL010',
                'SL015',
                'SL016',
                'SL018',
                'SL019',
                'SL026',
                'SL027',
                'SL028',
                'SL029',
                'SL034',
                'SL035',
                'SL036',
                'SL037',
                'SL039',
                'SL040',
                'SL042',
                'SL043',
                'SL047',
                'SL048',
                'SL049',
                'SL050',
                'SL051',
                'SL053',
                'SL055',
                'SL056',
                'SL057',
                'SL058',
                'SL059',
                'SL060',
                'SL061',
                'SL063',
                'SL064',
                'SL065',
                'SL066',
                'SL069',
                'SL070',
                'SL071',
                'SL072',
                'SL073',
                'SL074',
                'SL075',
                'SL076',
                'SL077',
                'SL078',
                'SL082',
                'SL086',
                'SL088',
                'SL089',
                'SL090',
                'SL091',
                'SL094',
                'SL095',
                'SL096',
                'SL099',
                'SL100',
                'SL103',
                'SL104',
                'SL105',
                'SL106',
                'SL109',
                'SL110',
                'SL111',
                'SL112',
                'SL115',
                'SL117',
                'SL118',
                'SL119',
                'SL120',
                'SL121',
                'SL123',
                'SL128',
                'SL129',
                'SL130',
                'SL131',
                'SL132',
                'SL133',
                'SL134',
                'SL135',
                'SL136',
                'SL137',
                'SL138',
                'SL141',
                'SL142',
                'SL144',
                'SL145',
                'SL147',
                'SL148',
                'SL151',
                'SL152',
                'SL153',
                'SL154',
                'SL156',
                'SL157',
                'SL158',
                'SL159',
                'SL161',
                'SL162',
                'SL163',
                'SL164',
                'SL165',
                'SL166',
                'SL167',
                'SL168',
                'SL170',
                'SL171',
                'SL173',
                'SL174',
                'SL175',
                'SL176',
                'SL177',
                'SL178',
                'SL179',
                'SL180',
                'SL182',
                'SL184',
                'SL185',
                'SL186',
                'SL187',
                'SL188',
                'SL190',
                'SL191',
                'SL192',
                'SL193',
                'SL194',
                'SL195',
                'SL196',
                'SL199',
                'SL200',
                'SL201',
                'SL207',
                'SL208',
                'SL209',
                'SL210',
                'SL213',
                'SL214',
                'SL217',
                'SL218',
                'SL219',
                'SL220',
                'SL224',
                'SL225',
                'SL226',
                'SL230',
                'SL231',
                'SL239',
                'SL241',
                'SL243',
                'SL244',
                'SL245',
                'SL247',
                'SL248',
                'SL250',
                'SL252',
                'SL253',
                'SL254',
                'SL255',
                'SL256',
                'SL257',
                'SL259',
                'SL261',
                'SL262',
                'SL264',
                'SL266',
                'SL267',
                'SL268',
                'SL269',
                'SL271',
                'SL274',
                'SL277',
                'SL278',
                'SL280',
                'SL281',
                'SL282',
                'SL283',
                'SL285',
                'SL287',
                'SL289',
                'SL291',
                'SL293',
                'SL294',
                'SL295',
                'SL296',
                'SL297',
                'SL298',
                'SL299',
                'SL300',
                'SL319',
                'SL320',
                'SL332',
                'SL336',
                'SL341',
                'SL348',
                'SL349',
                'SL358',
                'SL359',
                'SL361',
                'SL363',
                'SL364',
                'SL365',
                'SL366',
                'SL367',
                'SL369',
                'SL370',
                'SL371',
                'SL372',
                'SL373',
                'SL374',
                'SL375',
                'SL378',
                'SL379',
                'SL380',
                'SL381',
                'SL382',
                'SL383',
                'SL384',
                'SL385',
                'SL386',
                'SL387',
                'SL388',
                'SL389',
                'SL390',
                'SL392',
                'SL393',
                'SL394',
                'SL395',
                'SL396',
                'SL397',
                'SL398',
                'SL400',
                'SL401',
                'SL403',
                'SL404',
                'SL406',
                'SL409',
                'SL412',
                'SL414',
                'SL415',
                'SL416',
                'SL418',
                'SL419',
                'SL420',
                'SL421',
                'SL422',
                'SL423',
                'SL425',
                'SL426',
                'SL427',
                'SL428',
                'SL430',
                'SL431',
                'SL432',
                'SL433',
                'SL434',
                'SL435',
                'SL436',
                'SL437',
                'SL438',
                'SL440',
                'SL441',
                'SL442',
                'SL443',
                'SL444',
                'SL445',
                'SL446',
                'SL447',
                'SL448',
                'SL449',
                'SL450',
                'SL451',
                'SL452',
                'SL453',
                'SL454',
                'SL455',
                'SL456',
                'SL457',
                'SL458',
                'SL460',
                'SL462',
                'SL463',
                'SL464',
                'SL465',
                'SL467',
                'SL468',
                'SL469',
                'SL470',
                'SL471',
                'SL472',
                'SL473',
                'SL475',
                'SL476',
                'SL478',
                'SL479',
                'SL480',
                'SL481',
                'SL482',
                'SL483',
                'SL485',
                'SL487',
                'SL488',
                'SL489',
                'SL490',
                'SL491',
                'SL493',
                'SL494',
                'SL495',
                'SL496',
                'SL498',
                'SL500',
                'SL501',
                'SL502',
                'SL504',
                'SL505',
                'SL506',
                'SL508',
                'SL510',
                'SL511',
                'SL512',
                'SL513',
                'SL514',
                'SL515',
                'SL516'],
 'traits': ['SlFM0030',
            'SlFM0047',
            'SlFM0048',
            'SlFM0069',
            'SlFM0216',
            'SlFM0222',
            'SlFM0244',
            'SlFM0259',
            'SlFM0275',
            'SlFM0336',
            'SlFM0346',
            'SlFM0386',
            'SlFM0416',
            'SlFM0421',
            'SlFM0430',
            'SlFM0438',
            'SlFM0502',
            'SlFM0533',
            'SlFM0555',
            'SlFM0577',
            'SlFM0625',
            'SlFM0716',
            'SlFM0778',
            'SlFM0982',
            'SlFM1384',
            'SlFM1470',
            'SlFM1542',
            'SlFM1553',
            'SlFM1619',
            'SlFM1621',
            'SlFM1804',
            'SlFM1826',
            'SlFM1952',
            'SlFM1980',
            'SlFM1984',
            'SlFM1988'],
 'fold_partitions': [{'fold': 0,
                      'train': [0,
                                1,
                                2,
                                3,
                                4,
                                6,
                                8,
                                10,
                                11,
                                12,
                                14,
                                15,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25,
                                26,
                                27,
                                28,
                                29,
                                30,
                                32,
                                33,
                                34,
                                35,
                                36,
                                37,
                                38,
                                39,
                                41,
                                42,
                                43,
                                44,
                                45,
                                46,
                                47,
                                48,
                                49,
                                50,
                                51,
                                52,
                                53,
                                54,
                                55,
                                56,
                                57,
                                58,
                                59,
                                60,
                                61,
                                62,
                                64,
                                66,
                                67,
                                68,
                                69,
                                70,
                                71,
                                72,
                                74,
                                75,
                                78,
                                79,
                                80,
                                81,
                                82,
                                83,
                                84,
                                85,
                                86,
                                87,
                                89,
                                91,
                                92,
                                93,
                                94,
                                95,
                                96,
                                97,
                                98,
                                99,
                                100,
                                101,
                                102,
                                103,
                                104,
                                106,
                                107,
                                108,
                                109,
                                110,
                                111,
                                112,
                                113,
                                114,
                                115,
                                118,
                                122,
                                123,
                                125,
                                126,
                                127,
                                128,
                                129,
                                130,
                                132,
                                133,
                                134,
                                135,
                                136,
                                137,
                                138,
                                139,
                                140,
                                142,
                                143,
                                144,
                                145,
                                146,
                                147,
                                148,
                                149,
                                150,
                                151,
                                152,
                                153,
                                154,
                                155,
                                156,
                                157,
                                158,
                                160,
                                161,
                                162,
                                163,
                                164,
                                165,
                                166,
                                168,
                                169,
                                170,
                                171,
                                172,
                                173,
                                176,
                                177,
                                180,
                                181,
                                182,
                                183,
                                184,
                                186,
                                188,
                                189,
                                190,
                                191,
                                192,
                                194,
                                195,
                                196,
                                197,
                                198,
                                199,
                                201,
                                202,
                                203,
                                204,
                                205,
                                206,
                                207,
                                209,
                                213,
                                217,
                                218,
                                219,
                                220,
                                222,
                                223,
                                224,
                                225,
                                227,
                                228,
                                230,
                                231,
                                232,
                                233,
                                236,
                                238,
                                239,
                                240,
                                241,
                                242,
                                243,
                                244,
                                246,
                                250,
                                252,
                                253,
                                254,
                                257,
                                258,
                                259,
                                260,
                                261,
                                262,
                                263,
                                264,
                                265,
                                267,
                                268,
                                269,
                                270,
                                271,
                                272,
                                273,
                                275,
                                276,
                                277,
                                278,
                                279,
                                281,
                                283,
                                284,
                                285,
                                286,
                                287,
                                288,
                                290,
                                291,
                                292,
                                293,
                                294,
                                295,
                                297,
                                298,
                                299,
                                300,
                                301,
                                302,
                                303,
                                304,
                                305,
                                307,
                                310,
                                313,
                                314,
                                315,
                                316,
                                317,
                                318,
                                319,
                                320,
                                321,
                                322],
                      'held': [5,
                               7,
                               9,
                               13,
                               16,
                               31,
                               40,
                               63,
                               65,
                               73,
                               76,
                               77,
                               88,
                               90,
                               105,
                               116,
                               117,
                               119,
                               120,
                               121,
                               124,
                               131,
                               141,
                               159,
                               167,
                               174,
                               175,
                               178,
                               179,
                               185,
                               187,
                               193,
                               200,
                               208,
                               210,
                               211,
                               212,
                               214,
                               215,
                               216,
                               221,
                               226,
                               229,
                               234,
                               235,
                               237,
                               245,
                               247,
                               248,
                               249,
                               251,
                               255,
                               256,
                               266,
                               274,
                               280,
                               282,
                               289,
                               296,
                               306,
                               308,
                               309,
                               311,
                               312,
                               323],
                      'inner_train': [0,
                                      1,
                                      3,
                                      4,
                                      8,
                                      10,
                                      11,
                                      12,
                                      14,
                                      15,
                                      17,
                                      18,
                                      19,
                                      20,
                                      21,
                                      23,
                                      25,
                                      26,
                                      27,
                                      29,
                                      30,
                                      32,
                                      33,
                                      34,
                                      35,
                                      36,
                                      37,
                                      38,
                                      39,
                                      42,
                                      43,
                                      45,
                                      46,
                                      49,
                                      50,
                                      51,
                                      54,
                                      55,
                                      58,
                                      61,
                                      62,
                                      66,
                                      67,
                                      68,
                                      69,
                                      70,
                                      71,
                                      72,
                                      74,
                                      78,
                                      80,
                                      81,
                                      82,
                                      83,
                                      84,
                                      86,
                                      87,
                                      89,
                                      91,
                                      92,
                                      94,
                                      95,
                                      96,
                                      98,
                                      99,
                                      100,
                                      101,
                                      102,
                                      103,
                                      106,
                                      107,
                                      108,
                                      109,
                                      110,
                                      111,
                                      112,
                                      113,
                                      114,
                                      115,
                                      118,
                                      122,
                                      123,
                                      125,
                                      126,
                                      127,
                                      128,
                                      129,
                                      132,
                                      133,
                                      134,
                                      135,
                                      136,
                                      137,
                                      138,
                                      139,
                                      142,
                                      145,
                                      147,
                                      148,
                                      149,
                                      150,
                                      151,
                                      152,
                                      156,
                                      157,
                                      158,
                                      160,
                                      161,
                                      163,
                                      164,
                                      165,
                                      166,
                                      168,
                                      169,
                                      170,
                                      171,
                                      173,
                                      176,
                                      181,
                                      182,
                                      183,
                                      186,
                                      188,
                                      189,
                                      190,
                                      191,
                                      192,
                                      195,
                                      198,
                                      199,
                                      201,
                                      202,
                                      203,
                                      204,
                                      205,
                                      207,
                                      209,
                                      213,
                                      217,
                                      218,
                                      219,
                                      220,
                                      222,
                                      223,
                                      225,
                                      227,
                                      228,
                                      230,
                                      231,
                                      232,
                                      233,
                                      236,
                                      239,
                                      240,
                                      241,
                                      243,
                                      244,
                                      246,
                                      250,
                                      252,
                                      253,
                                      254,
                                      258,
                                      259,
                                      260,
                                      261,
                                      262,
                                      264,
                                      265,
                                      267,
                                      268,
                                      269,
                                      270,
                                      271,
                                      272,
                                      277,
                                      278,
                                      279,
                                      281,
                                      284,
                                      285,
                                      286,
                                      287,
                                      288,
                                      290,
                                      291,
                                      292,
                                      293,
                                      295,
                                      298,
                                      299,
                                      300,
                                      301,
                                      302,
                                      303,
                                      305,
                                      307,
                                      310,
                                      313,
                                      315,
                                      316,
                                      317,
                                      318,
                                      319,
                                      320,
                                      321,
                                      322],
                      'inner_val': [2,
                                    6,
                                    22,
                                    24,
                                    28,
                                    41,
                                    44,
                                    47,
                                    48,
                                    52,
                                    53,
                                    56,
                                    57,
                                    59,
                                    60,
                                    64,
                                    75,
                                    79,
                                    85,
                                    93,
                                    97,
                                    104,
                                    130,
                                    140,
                                    143,
                                    144,
                                    146,
                                    153,
                                    154,
                                    155,
                                    162,
                                    172,
                                    177,
                                    180,
                                    184,
                                    194,
                                    196,
                                    197,
                                    206,
                                    224,
                                    238,
                                    242,
                                    257,
                                    263,
                                    273,
                                    275,
                                    276,
                                    283,
                                    294,
                                    297,
                                    304,
                                    314],
                      'training_seed': 20261012}],
 'primary_aggregation': 'Equal-weight mean across the 36 traits on the single fixed held-out partition (fold '
                        '0)',
 'uncertainty': 'No replicate-based interval is estimated from this single fixed split'}


ARCHITECTURES = {
    model['id']: {key: model[key] for key in ('width', 'depth', 'parameters')}
    for model in _BENCHMARK['models']
}


def get_benchmark():
    return deepcopy(_BENCHMARK)



def learning_rate(epoch, cfg):
    if epoch <= cfg['warmup']:
        return cfg['peak_lr'] * epoch / cfg['warmup']
    return cfg['peak_lr'] * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['warmup'])


def configure_device(device):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu or install CUDA-enabled PyTorch')
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def train_case(raw, metadata, accessions, outer_labels, fold, trait, model_cfg, training_cfg, device,
               save_optimizer=False):
    """Receive only this trait's outer-training labels; held labels never enter fitting."""
    device = torch.device(device)
    started = time.monotonic()
    tr, held = fold['train'], fold['held']
    inner, validation = fold['inner_train'], fold['inner_val']
    outer_y = np.asarray(outer_labels, dtype=np.float64)
    if outer_y.shape != (len(tr),) or not np.isfinite(outer_y).all():
        raise ValueError('Expected finite labels in outer-training accession order')
    if set(tr) & set(held) or set(inner) & set(validation) or set(inner) | set(validation) != set(tr):
        raise ValueError('Invalid train, validation, or held accession partition')
    if set(tr) | set(held) != set(range(len(raw))):
        raise ValueError('Outer partitions must cover the input accessions')
    position = {index: i for i, index in enumerate(tr)}
    mu, sd = float(outer_y.mean()), float(outer_y.std(ddof=1))
    if not sd > 0:
        raise ValueError('Outer-training labels have zero variance')
    cfg = training_cfg
    architecture = dict(cfg, width=model_cfg['width'], depth=model_cfg['depth'])

    def make_model(fitting_indices):
        y = outer_y[[position[i] for i in fitting_indices]]
        transform = TrainOnlyTransform().fit(raw[fitting_indices], y, metadata)
        torch.manual_seed(fold['training_seed'])
        np.random.seed(fold['training_seed'] % 2 ** 32)
        net = SVDKL(transform, architecture).to(device)
        parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        if parameters != model_cfg['parameters']:
            raise ValueError(f'Parameter count differs: {parameters} versus {model_cfg["parameters"]}')
        inputs = torch.as_tensor(transform.train_encoded, device=device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['peak_lr'], weight_decay=cfg['weight_decay'])
        return transform, net, optimizer, inputs, y

    def step(net, optimizer, inputs, epoch):
        lr = learning_rate(epoch, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = net.nll(inputs)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite GP NLL')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()), float(gradient), lr

    def report(phase, epoch):
        if epoch % 25 == 0:
            print(json.dumps({'model': model_cfg['id'], 'fold': fold['fold'], 'trait': trait,
                              'phase': phase, 'epoch': epoch}), flush=True)

    transform, net, optimizer, inputs, inner_y = make_model(inner)
    y_validation = (outer_y[[position[i] for i in validation]] - float(inner_y.mean())) / float(inner_y.std(ddof=1))
    validation_x = torch.as_tensor(transform.transform(raw[validation]), device=device)
    curve, best, best_epoch = [], float('inf'), 0
    for epoch in range(1, cfg['max_epochs'] + 1):
        loss, gradient, lr = step(net, optimizer, inputs, epoch)
        net.eval()
        with torch.no_grad():
            prediction = net(validation_x).cpu().numpy()
        error = float(np.mean((prediction - y_validation) ** 2))
        if not np.isfinite(error):
            raise FloatingPointError('Nonfinite inner validation NMSE')
        curve.append({'epoch': epoch, 'gp_nll': loss, 'validation_nmse': error,
                      'learning_rate': lr, 'gradient_norm_before_clip': gradient})
        if error < best:
            best, best_epoch = error, epoch
        elif epoch - best_epoch >= cfg['patience']:
            break
        report('inner_selection', epoch)
    del net, optimizer, inputs, validation_x, transform
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Fresh initialization, new preprocessing, and the same original LR schedule.
    transform, net, optimizer, inputs, final_y = make_model(tr)
    np.testing.assert_array_equal(final_y, outer_y)
    initial = net.transformer.embed.weight.detach().flatten()[:128].cpu().clone()
    refit_curve = []
    for epoch in range(1, best_epoch + 1):
        loss, gradient, lr = step(net, optimizer, inputs, epoch)
        refit_curve.append({'epoch': epoch, 'gp_nll': loss, 'learning_rate': lr,
                            'gradient_norm_before_clip': gradient})
        report('outer_refit', epoch)
    parameter_change = float((net.transformer.embed.weight.detach().flatten()[:128].cpu() - initial).abs().max())
    if not parameter_change > 0:
        raise RuntimeError('Model parameters did not change during refit')
    net.eval()
    full = full_checkpoint(net, transform, model_cfg, cfg, {
        'model': model_cfg['id'], 'trait': trait, 'fold': fold['fold'],
        'accessions': list(accessions), 'training_seed': fold['training_seed'],
        'best_epoch': best_epoch, 'refit_epochs': best_epoch,
        'train_ids': [accessions[i] for i in tr], 'held_ids': [accessions[i] for i in held],
        'inner_train_ids': [accessions[i] for i in inner],
        'inner_val_ids': [accessions[i] for i in validation],
        'test_labels_in_fit_payload': False,
    }, optimizer=optimizer if save_optimizer else None)
    with torch.no_grad():
        held_x = torch.as_tensor(transform.transform(raw[held]), device=device)
        before = net(held_x).cpu().numpy().astype(np.float64) * sd + mu
        net.freeze()
        prediction = net(held_x).cpu().numpy().astype(np.float64) * sd + mu
    difference = float(np.max(np.abs(before - prediction)))
    if difference > 1e-3 * sd:
        raise RuntimeError('Frozen inference changed predictions')
    state = net.compact_state(transform)
    state.update({
        'model': np.asarray(model_cfg['id']), 'trait': np.asarray(trait), 'fold': np.asarray(fold['fold']),
        'accessions': np.asarray(accessions, dtype='U'),
        'train_ids': np.asarray([accessions[i] for i in tr], dtype='U'),
        'held_ids': np.asarray([accessions[i] for i in held], dtype='U'),
        'training_seed': np.asarray(fold['training_seed']), 'best_epoch': np.asarray(best_epoch),
    })
    restored = predict_values(state, raw[held][:, state['marker_cols']])
    restore_difference = float(np.max(np.abs(restored - prediction)) / sd)
    if restore_difference >= 2e-4:
        raise RuntimeError(f'Compact predictor restoration differs by {restore_difference} training SD')
    details = {
        'best_epoch': best_epoch, 'inner_best_nmse': best, 'inner_epochs_run': len(curve), 'curve': curve,
        'refit_curve': refit_curve, 'refit_epochs': best_epoch, 'parameter_change_first128': parameter_change,
        'fresh_refit': True, 'outer_test_used_for_selection': False, 'freeze_max_difference': difference,
        'compact_restore_max_difference_over_train_sd': restore_difference,
    }
    stem = f'{model_cfg["id"]}_f{fold["fold"]}_{trait}'
    row = {
        'model': model_cfg['id'], 'fold': fold['fold'], 'trait': trait, 'n_parameters': model_cfg['parameters'],
        'train_ids': [accessions[i] for i in tr], 'held_ids': [accessions[i] for i in held],
        'inner_train_ids': [accessions[i] for i in inner], 'inner_val_ids': [accessions[i] for i in validation],
        'n_train': len(tr), 'n_held': len(held), 'n_markers': len(transform.marker_cols),
        'train_mean': mu, 'train_sd_ddof1': sd, 'predictions': prediction.tolist(),
        'training_seed': fold['training_seed'], 'checkpoint': stem + '.npz', 'training': details,
        'full_checkpoint': stem + '.full.pt', 'optimizer_state_saved': save_optimizer,
        'seconds': time.monotonic() - started, 'device': str(device), 'test_labels_in_fit_payload': False,
    }
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite held predictions')
    # Retain the fitted GPU inference coefficients alongside the complete,
    # unfrozen network. CPU/GPU Transformer kernels can differ numerically;
    # these actual fitted coefficients make saved prediction portable without
    # changing the training procedure or relaxing the reload tolerance.
    full['inference_state'] = {
        key: torch.from_numpy(np.asarray(state[key]).copy())
        for key in ('marker_weights', 'alpha', 'base_coefficient')
    }
    # Keep the actual curves and case record even if a later validation fails.
    full['training_record'] = json.loads(json.dumps(row, allow_nan=False))
    return row, state, full


def save_full_checkpoint(path, payload):
    """Write the full checkpoint atomically without replacing an existing artifact."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pending', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def validate_full_case(path, record, raw, metadata, fold, require_optimizer=False):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f'Full trained network is missing: {path}. Existing compact predictions cannot '
            'reconstruct historical Transformer parameters. Keep the existing results; '
            'a newly trained full network requires a separate output directory.')
    net, transform, payload, _ = load_full_checkpoint(path)
    for key in ('model', 'trait', 'fold', 'training_seed', 'train_ids', 'held_ids',
                'inner_train_ids', 'inner_val_ids', 'test_labels_in_fit_payload'):
        if payload['case'].get(key) != record.get(key):
            raise ValueError(f'Full checkpoint differs from its training record in {key}')
    if payload['case']['best_epoch'] != record['training']['best_epoch']:
        raise ValueError('Full-checkpoint refit epoch differs from its training record')
    if require_optimizer and not payload['optimizer_saved']:
        raise ValueError('Existing full checkpoint has no optimizer state; it cannot be recovered from predictions')
    np.testing.assert_array_equal(transform.input_marker_ids, metadata['marker_id'])
    prediction = predict_full_values(net, transform, raw[fold['held']], metadata['marker_id'])
    portable_difference = float(np.max(np.abs(prediction - np.asarray(record['predictions']))) / transform.sd)
    if portable_difference >= 2e-4:
        raise ValueError('Full checkpoint does not restore its recorded predictions')
    checks = {'full_checkpoint_portable_max_difference_over_train_sd': portable_difference}
    if 'inference_state' in payload:
        original_device = str(record.get('device', 'cpu'))
        if original_device.startswith('cuda') and not torch.cuda.is_available():
            if not record.get('training', {}).get('full_checkpoint_origin_device_reload_verified'):
                raise RuntimeError('Origin-device full-network reload must pass before CPU-only reuse')
            checks['full_checkpoint_origin_device_reload_reused'] = True
        else:
            origin_net, origin_transform, _, _ = load_full_checkpoint(path, device=original_device)
            # Recompute from the complete neural weights on their training device;
            # this deliberately does not use the saved inference cache.
            origin_net.freeze()
            encoded = torch.as_tensor(origin_transform.transform(raw[fold['held']]),
                                      device=next(origin_net.parameters()).device)
            with torch.no_grad():
                origin_prediction = origin_net(encoded).cpu().numpy().astype(np.float64)
            origin_prediction = origin_prediction * origin_transform.sd + origin_transform.mean
            origin_difference = float(np.max(np.abs(origin_prediction - np.asarray(record['predictions'])))
                                      / origin_transform.sd)
            if origin_difference >= 2e-4:
                raise ValueError('Serialized full neural weights do not restore their origin-device predictions')
            checks.update(full_checkpoint_origin_device_reload_verified=True,
                          full_checkpoint_origin_max_difference_over_train_sd=origin_difference)
    return checks


def validate_completed_case(record, state, checkpoint, fold, trait, model_cfg, cfg, raw, metadata, accessions):
    expected = {
        'model': model_cfg['id'], 'trait': trait, 'fold': fold['fold'],
        'n_parameters': model_cfg['parameters'], 'training_seed': fold['training_seed'],
        'train_ids': [accessions[i] for i in fold['train']],
        'held_ids': [accessions[i] for i in fold['held']],
        'inner_train_ids': [accessions[i] for i in fold['inner_train']],
        'inner_val_ids': [accessions[i] for i in fold['inner_val']],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'Existing case differs in {key}')
        if key in state and not np.array_equal(state[key], np.asarray(value)):
            raise ValueError(f'Existing checkpoint and record differ in {key}')
    details = record['training']
    if (details.get('fresh_refit') is not True or details.get('outer_test_used_for_selection') is not False
            or record.get('test_labels_in_fit_payload') is not False):
        raise ValueError('Existing case does not follow the fixed training protocol')
    best = details['best_epoch']
    curve, refit = details['curve'], details['refit_curve']
    if 'best_epoch' in state and int(state['best_epoch']) != best:
        raise ValueError('Existing checkpoint and record differ in best_epoch')
    if any(not np.isfinite(row['validation_nmse']) for row in curve):
        raise ValueError('Existing validation curve has non-finite values')
    if (not curve or best != details['refit_epochs'] or best != len(refit)
            or best != min(curve, key=lambda x: x['validation_nmse'])['epoch']
            or len(curve) > cfg['max_epochs']
            or details.get('inner_epochs_run') != len(curve)
            or (len(curve) != cfg['max_epochs'] and len(curve) != best + cfg['patience'])):
        raise ValueError('Existing case has inconsistent epoch records')
    for phase in (curve, refit):
        if [row['epoch'] for row in phase] != list(range(1, len(phase) + 1)):
            raise ValueError('Existing curve has missing epochs')
        for row in phase:
            if not np.isfinite(row['gp_nll']) or not np.isclose(row['learning_rate'], learning_rate(row['epoch'], cfg), rtol=0, atol=1e-15):
                raise ValueError('Existing case has an invalid loss or learning-rate schedule')
    np.testing.assert_array_equal(marker_catalog(state, checkpoint), metadata['marker_id'])
    np.testing.assert_allclose([float(state['mean']), float(state['sd'])],
                               [record['train_mean'], record['train_sd_ddof1']], rtol=0, atol=1e-12)
    if len(state['d_pos']) != cfg['top_k'] or len(state['marker_weights']) != cfg['top_k']:
        raise ValueError('Existing checkpoint does not contain the selected 500 markers')
    prediction = predict_values(state, raw[fold['held']][:, state['marker_cols']])
    saved = np.asarray(record['predictions'], dtype=float)
    if saved.shape != prediction.shape or not np.isfinite(saved).all():
        raise ValueError('Existing case has invalid held-out predictions')
    if np.max(np.abs(saved - prediction)) / float(state['sd']) >= 2e-4:
        raise ValueError('Existing checkpoint does not restore its recorded predictions')


def input_snapshot_path(root):
    root = Path(root)
    folders = [root]
    if root.name in ARCHITECTURES:
        folders.append(root.parent)
    elif root.name in {f'fold_{i}' for i in range(5)} and root.parent.name in ARCHITECTURES:
        folders.extend([root.parent, root.parent.parent])
    for folder in list(folders):
        if folder.name == 'rebuilt_fold0':
            folders.append(folder.parent)
    for folder in folders:
        snapshot = folder / '_input_snapshot.npz'
        if snapshot.exists() or snapshot.is_symlink():
            return snapshot
    return root / '_input_snapshot.npz'


def find_saved_case(root, model_id, fold, trait):
    root = Path(root)
    stem = f'{model_id}_f{fold}_{trait}'
    folders = dict.fromkeys((root / model_id / f'fold_{fold}', root / f'fold_{fold}',
                            root / model_id, root))
    occupied = []
    for folder in folders:
        paths = tuple(folder / (stem + suffix) for suffix in ('.json', '.npz'))
        if any(path.exists() or path.is_symlink() for path in paths):
            occupied.append(paths)
    if len(occupied) > 1:
        raise ValueError(f'Duplicate case locations for {stem}; keep one record/checkpoint pair')
    if not occupied:
        return None
    if not all(path.is_file() for path in occupied[0]):
        raise FileNotFoundError(f'Incomplete record/checkpoint pair for {stem}')
    return occupied[0]


def new_case_directory(root, model_id, fold):
    root = Path(root)
    if root.name in ARCHITECTURES:
        if root.name != model_id:
            raise ValueError('Output model directory differs from the selected model')
        return root / f'fold_{fold}'
    if root.name.startswith('fold_') and root.parent.name in ARCHITECTURES:
        if root.parent.name != model_id or root.name != f'fold_{fold}':
            raise ValueError('Output fold directory differs from the selected model/fold')
        return root
    return root / model_id / f'fold_{fold}'


def bind_input_snapshot(output, arrays):
    snapshot = input_snapshot_path(output)

    def compare_snapshot():
        try:
            with np.load(snapshot, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays):
                    raise ValueError('Input array names differ from the output snapshot')
                for key, current in arrays.items():
                    previous = saved[key]
                    same = (previous.shape == current.shape and previous.dtype == current.dtype)
                    if same:
                        same = np.array_equal(previous, current, equal_nan=True) if current.dtype.kind in 'fc' else np.array_equal(previous, current)
                    if not same:
                        raise ValueError(f'Input array {key} differs from the output snapshot; use a new output directory')
        except (OSError, EOFError, KeyError) as error:
            raise ValueError('Cannot read the output input snapshot') from error

    if snapshot.exists():
        compare_snapshot()
        return
    lock = snapshot.with_suffix('.lock')
    try:
        lease = lock.open('x')
    except FileExistsError:
        raise RuntimeError('Input snapshot initialization is in progress; retry after the other process finishes') from None
    temporary = None
    try:
        with lease:
            if snapshot.exists():
                compare_snapshot()
                return
            folder = snapshot.parent
            if any(path.suffix in ('.json', '.npz') for path in folder.rglob('SVDKL-*_f*_*.*')):
                raise ValueError('Existing cases have no input snapshot; input compatibility cannot be established')
            with tempfile.NamedTemporaryFile(dir=folder, prefix='_input_snapshot_', suffix='.pending', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, snapshot)
    finally:
        if temporary is not None and temporary.is_file():
            temporary.unlink()
        lock.unlink(missing_ok=True)


def validate_plan(plan, model_id):
    matches = [model for model in plan.get('models', []) if model.get('id') == model_id]
    if len(matches) != 1 or model_id not in ARCHITECTURES:
        raise ValueError('Configuration must contain exactly one entry for the selected SVDKL model')
    model_cfg = matches[0]
    if any(type(model_cfg.get(key)) is not int or model_cfg[key] != value
           for key, value in ARCHITECTURES[model_id].items()):
        raise ValueError('Model width, depth and parameter count must match its fixed SVDKL capacity')
    fold_ids = [fold.get('fold') for fold in plan.get('fold_partitions', [])]
    if fold_ids != [0] or any(type(value) is not int for value in fold_ids):
        raise ValueError('Configuration must contain only the fixed split with fold ID 0')
    if plan['fold_partitions'] != _BENCHMARK['fold_partitions']:
        raise ValueError('The fixed accession roles or training seed were changed')
    cfg = plan['svdkl']
    required = {'max_epochs': 105, 'warmup': 10, 'peak_lr': 0.005, 'patience': 45,
                'heads': 4, 'dropout': 0.1, 'beta_init': 0.1, 'delta_init': 1.0,
                'weight_decay': 0.0, 'grad_clip': 1.0, 'top_k': 500, 'optimizer': 'AdamW', 'schedule': 'linear'}
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('Benchmark settings differ from the fixed SVDKL training protocol')
    return model_cfg, cfg


def main():
    parser = argparse.ArgumentParser(description='Train SVDKL once per trait on the fixed fold-0 split.')
    parser.add_argument('--model', default='SVDKL-0.01M', choices=list(ARCHITECTURES))
    parser.add_argument('--trait', default='all', help='One trait ID or all')
    parser.add_argument('--fold', default='0', choices=['0'], help='The published fixed split (0)')
    parser.add_argument('--device', default='cuda:0', help='cpu or cuda:N')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--data', type=Path, default=HERE / 'data' / 'dataset_inputs.npz')
    parser.add_argument('--resume', action='store_true', help='Compare all inputs with the output snapshot, then verify and skip complete cases')
    parser.add_argument('--save-optimizer', action='store_true',
                        help='Also save final AdamW state in the full checkpoint; increases disk use')
    args = parser.parse_args()
    device = None
    plan = get_benchmark()
    model_cfg, cfg = validate_plan(plan, args.model)
    with np.load(args.data, allow_pickle=False) as data:
        input_arrays = {key: data[key] for key in data.files}
    raw, labels = input_arrays['dosage'], input_arrays['phenotypes']
    accessions, traits = input_arrays['accessions'].tolist(), input_arrays['traits'].tolist()
    metadata = {k: input_arrays[k] for k in ('marker_id', 'chrom', 'pos', 'svlen')}
    if 'svtype' in input_arrays:
        metadata['svtype'] = input_arrays['svtype']
    if accessions != plan['accessions'] or traits != plan['traits']:
        raise ValueError('Dataset accession/trait order differs from the fixed benchmark')
    if raw.shape != (324, 106362) or labels.shape != (324, 36):
        raise ValueError('Expected the fixed 324-material, 36-trait, 106362-marker dataset')
    if len(set(accessions)) != len(accessions) or len(set(traits)) != len(traits):
        raise ValueError('Duplicate accession or trait IDs')
    folds = plan['fold_partitions']
    if (len(folds) != 1 or folds[0]['fold'] != 0
            or set(folds[0]['train']) & set(folds[0]['held'])
            or sorted(folds[0]['train'] + folds[0]['held']) != list(range(len(raw)))):
        raise ValueError('The fixed training and held-out roles must partition the dataset')
    if args.trait != 'all' and args.trait not in traits:
        raise ValueError(f'Unknown trait: {args.trait}')
    selected_traits = traits if args.trait == 'all' else [args.trait]
    selected_folds = folds
    args.output.mkdir(parents=True, exist_ok=True)
    bind_input_snapshot(args.output, input_arrays)
    with threadpool_limits(limits=1):
        for fold in selected_folds:
            for trait in selected_traits:
                stem = f'{args.model}_f{fold["fold"]}_{trait}'
                paths = find_saved_case(args.output, args.model, fold['fold'], trait)
                if paths is not None:
                    if args.resume:
                        record = json.loads(paths[0].read_text(encoding='utf-8'))
                        state = load_checkpoint(paths[1])
                        validate_full_case(paths[1].with_suffix('.full.pt'), record, raw, metadata, fold,
                                           require_optimizer=args.save_optimizer)
                        validate_completed_case(record, state, paths[1], fold, trait, model_cfg, cfg, raw, metadata, accessions)
                        print(json.dumps({'verified_and_skipped': stem}), flush=True)
                        continue
                    raise FileExistsError(f'Output already exists for {stem}; choose a new --output directory')
                case_output = new_case_directory(args.output, args.model, fold['fold'])
                case_output.mkdir(parents=True, exist_ok=True)
                paths = [case_output / (stem + suffix) for suffix in ('.json', '.npz')]
                full_path = case_output / (stem + '.full.pt')
                if full_path.exists() or full_path.is_symlink():
                    raise FileExistsError(f'Full checkpoint exists without a complete result pair: {full_path}')
                # Only outer-training labels are passed into the fitting function.
                if device is None:
                    device = configure_device(args.device)
                outer_labels = labels[fold['train'], traits.index(trait)].copy()
                row, state, full = train_case(raw, metadata, accessions, outer_labels, fold, trait,
                                              model_cfg, cfg, device, save_optimizer=args.save_optimizer)
                save_full_checkpoint(full_path, full)
                reload_checks = validate_full_case(full_path, row, raw, metadata, fold,
                                                   require_optimizer=args.save_optimizer)
                row['training'].update(reload_checks)
                row['training']['full_checkpoint_disk_reload_verified'] = True
                np.savez_compressed(paths[1], **state)
                paths[0].write_text(json.dumps(row, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                print(json.dumps({'completed': stem, 'best_epoch': row['training']['best_epoch'],
                                  'seconds': row['seconds']}), flush=True)
                del state, full
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
