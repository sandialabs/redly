#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.tx_utils as tx_utils
from egret.parsers.matpower_parser import create_ModelData
import tensorflow as tf
from tensorflow.data import Dataset
from math import radians


class ACPFDataset(object):
    '''
    Generic ACPF dataset class with common utility functions
    ''' 
    def __init__(self, casefile, b_p=1., constraints=None):
        self.casefile = casefile
        self.md = create_ModelData(casefile['mfile'])
        self.b_p = b_p
        self.constraints = constraints
        
        # load data
        # generates self.incols, self.inputs, self.outcols, self.outputs, self.cycles
        if casefile['type']=='egret':
            self.load_egret()
        elif casefile['type']=='pw':
            self.load_pw()
        
        # build model_data dictionary
        self.get_model_data()

        # build test constraints if none are specified
        # *note* this leaks test statistics to downstream NN
        if not constraints:
            self.build_test_constraints()
            
        # sample boundary/collocation pts
        self.b_flags = (np.random.random(len(self.inputs[0]))
                        <b_p).astype(np.float32).reshape(-1,1)/b_p

    def load_egret(self):
        '''
        Function to load and parse Egret data (rev 2.0)
        '''
        indf = pd.read_csv(self.casefile['infile'])
        outdf = pd.read_csv(self.casefile['outfile'])
        cdf = pd.read_csv(self.casefile['cfile'])
        
        # Some cases have an additional index column
        indf = indf[indf.columns[1:]]
        outdf = outdf[outdf.columns[1:]]
        
        incols, outcols = np.array(indf.columns), np.array(outdf.columns)
        
        # generators
        pgcols = []
        pgind = []
        qgcols = []
        qgind = []
        prcols = []
        prind = []
        gens = dict(self.md.elements(element_type='generator'))
        for gen in gens:
            if gen == self.casefile['slack']:
                prcols.append('pg_b_%s_g_%s'%(gens[gen]['bus'], gen))
                prind.append(np.where(np.array([col.startswith('pg:') for col in outcols])&
                                      np.array([col.endswith(',g%s'%(gen)) for col in outcols]))[0][0])
            else:
                if not gens[gen]['pg']==0: # don't count synchrophasors for pg
                    pgcols.append('pg_b_%s_g_%s'%(gens[gen]['bus'], gen))
                    pgind.append(np.where(np.array([col.startswith('pg:') for col in incols])&
                                          np.array([col.endswith(',g%s'%(gen)) for col in incols]))[0][0])
            qgcols.append('qg_b_%s_g_%s'%(gens[gen]['bus'], gen))
            qgind.append(np.where(np.array([col.startswith('qg:') for col in outcols])&
                                  np.array([col.endswith(',g%s'%(gen)) for col in outcols]))[0][0])
        pgdat = indf.values[:,pgind]/self.casefile['scale']
        qgdat = outdf.values[:,qgind]/self.casefile['scale']
        prdat = outdf.values[:,prind]/self.casefile['scale']
        
        # loads
        plcols = []
        plind = []
        qlcols = []
        qlind = []
        loads = dict(self.md.elements(element_type='load'))
        for load in loads:
            bus = loads[load]['bus']
            if 'pl:b%s'%(bus) in incols:
                plcols.append('pl_b_%s'%(bus))
                plind.append(np.where(np.array([col=='pl:b%s'%(bus) for col in incols]))[0][0])
            if 'ql:b%s'%(bus) in incols:
                qlcols.append('ql_b_%s'%(bus))
                qlind.append(np.where(np.array([col=='ql:b%s'%(bus) for col in incols]))[0][0])
        pldat = indf.values[:,plind]/self.casefile['scale']
        qldat = indf.values[:,qlind]/self.casefile['scale']
            
        # voltages
        wicols = []
        wiind = []
        wocols = []
        woind = []
        buses = dict(self.md.elements(element_type='bus'))
        for bus in buses:
            if buses[bus]['matpower_bustype']=='PQ':  # non-generator bus
                wocols.append('w_b_%s'%(bus))
                woind.append(np.where(outcols=='vm:b%s'%(bus))[0][0])
            elif bus==self.casefile['ref']:
                wicols.append('w_b_%s'%(bus))
                if 'mod' in self.casefile:
                    wiind.append(np.where(incols=='vm_ref:%s'%(bus))[0][0])
                else:
                    wiind.append(np.where(incols=='vm:%s,ref'%(bus))[0][0])
            else:
                wicols.append('w_b_%s'%(bus))
                wiind.append(np.where(incols=='vm:b%s'%(bus))[0][0])
        widat = indf.values[:,wiind]**2
        wodat = outdf.values[:,woind]**2
                
        # c, s
        ccols = []
        scols = []
        vmfind = []
        vafind = []
        vmtind = []
        vatind = []
        tdat = np.concatenate([indf.values,outdf.values],axis=1)
        tcols = np.concatenate([incols, outcols])
        branches = dict(self.md.elements(element_type='branch'))  # assumes one line/branch
        for branch in branches:
            bf = branches[branch]['from_bus']
            bt = branches[branch]['to_bus']
            ccols.append('c_f_%s_t_%s'%(bf,bt))
            scols.append('s_f_%s_t_%s'%(bf,bt))
            if bf==self.casefile['ref']:
                if 'mod' in self.casefile:
                    vmfind.append(np.where(tcols=='vm_ref:%s'%(bf))[0][0])
                    vafind.append(np.where(tcols=='va_ref:%s'%(bf))[0][0])
                else:
                    vmfind.append(np.where(tcols=='vm:%s,ref'%(bf))[0][0])
                    vafind.append(np.where(tcols=='va:%s,ref'%(bf))[0][0])
            else:
                vmfind.append(np.where(tcols=='vm:b%s'%(bf))[0][0])
                vafind.append(np.where(tcols=='va:%s'%(bf))[0][0])
            if bt==self.casefile['ref']:
                if 'mod' in self.casefile:
                    vmtind.append(np.where(tcols=='vm_ref:%s'%(bt))[0][0])
                    vatind.append(np.where(tcols=='va_ref:%s'%(bt))[0][0])
                else:
                    vmtind.append(np.where(tcols=='vm:%s,ref'%(bt))[0][0])
                    vatind.append(np.where(tcols=='va:%s,ref'%(bt))[0][0])
            else:
                vmtind.append(np.where(tcols=='vm:b%s'%(bt))[0][0])
                vatind.append(np.where(tcols=='va:%s'%(bt))[0][0])
        vmfdat = tdat[:,vmfind]
        vafdat = tdat[:,vafind]
        vmtdat = tdat[:,vmtind]
        vatdat = tdat[:,vatind]
        cdat = vmfdat*vmtdat*np.cos(vafdat-vatdat)
        sdat = vmfdat*vmtdat*np.sin(vafdat-vatdat)
        
        # cycles
        self.cycles = {}
        for i, col in enumerate(cdf.columns):
            angles = cdf[col].dropna().values
            tangles = []
            for angle in angles:
                bf = angle.split('b')[1].split('_')[0]
                bt = angle.split('b')[2]
                sign = '-' if angle.startswith('-') else '+'
                tangles.append('%scy_f_%s_t_%s'%(sign,bf,bt))
            self.cycles[str(i)] = tangles
            
        # indices are easier to handle at the end
        ioffsets = np.cumsum([0, len(pgcols), len(plcols), len(qlcols)])
        pginds = np.arange(len(pgcols)) + ioffsets[0]
        plinds = np.arange(len(plcols)) + ioffsets[1]
        qlinds = np.arange(len(qlcols)) + ioffsets[2]
        wiinds = np.arange(len(wicols)) + ioffsets[3]
        
        ooffsets = np.cumsum([0, len(prcols), len(qgcols), len(wocols), len(ccols)])
        prinds = np.arange(len(prcols)) + ooffsets[0]
        qginds = np.arange(len(qgcols)) + ooffsets[1]
        woinds = np.arange(len(wocols)) + ooffsets[2]
        cinds = np.arange(len(ccols)) + ooffsets[3]
        sinds = np.arange(len(scols)) + ooffsets[4]
        
        self.incols = [pgcols, plcols, qlcols, wicols]
        self.iinds = [pginds, plinds, qlinds, wiinds]
        self.inputs = [pgdat.astype(np.float32), pldat.astype(np.float32), 
                       qldat.astype(np.float32), widat.astype(np.float32)]
        self.outcols = [prcols, qgcols, wocols, ccols, scols]
        self.oinds = [prinds, qginds, woinds, cinds, sinds]
        self.outputs = [prdat.astype(np.float32), qgdat.astype(np.float32), 
                        wodat.astype(np.float32), cdat.astype(np.float32), sdat.astype(np.float32)]
    
    def load_pw(self):
        '''
        Function to load and parse PowerWorld data
        '''
        self.cycles = {}
        self.incols = []
        self.inputs = []
        self.outcols = []
        self.outputs = []
        
    def get_model_data(self, mscale=100):
        '''
        Function to build model_data file from modeldata structure and 
        preprocessed columns
        '''        
        self.model_data = {}
        
        pgcols, plcols, qlcols, wicols = self.incols
        prcols, qgcols, wocols, ccols, scols = self.outcols        
        # construct flowcols for now
        pfcols, ptcols, qfcols, qtcols = [[x+'_'+c[2:] for c in ccols] 
                                          for x in ['pf', 'pt', 'qf', 'qt']]
        flowcols = [pfcols, ptcols, qfcols, qtcols]
        
        self.model_data['incols'] = self.incols
        self.model_data['iinds'] = self.iinds
        self.model_data['outcols'] = self.outcols
        self.model_data['oinds'] = self.oinds
        self.model_data['flowcols'] = flowcols
        
        # flow matrices
        # real: [wi, wo, c, s] * K_pf = pf
        #       [wi, wo, c, s] * K_pt = pt
        # reac: [wi, wo, c, s] * K_qf = qf
        #       [wi, wo, c, s] * K_qt = qt
        # TODO: use sparse matrix structures
        cols_f = np.concatenate([wicols, wocols, ccols, scols])
        K_pf = np.zeros((len(cols_f), len(pfcols)), dtype=np.float32)
        K_pt = np.zeros((len(cols_f), len(ptcols)), dtype=np.float32)
        K_qf = np.zeros((len(cols_f), len(qfcols)), dtype=np.float32)
        K_qt = np.zeros((len(cols_f), len(qtcols)), dtype=np.float32)

        branches = dict(self.md.elements(element_type='branch'))
        for brkey in branches:
            branch = branches[brkey]
            bf = branch['from_bus']
            bt = branch['to_bus']
            br = np.where(np.array(pfcols)=='pf_f_%s_t_%s'%(bf,bt))[0][0]
            wcolf = np.where(cols_f=='w_b_%s'%(bf))[0][0]
            wcolt = np.where(cols_f=='w_b_%s'%(bt))[0][0]
            ccol = np.where(cols_f=='c_f_%s_t_%s'%(bf,bt))[0][0]
            scol = np.where(cols_f=='s_f_%s_t_%s'%(bf,bt))[0][0]
            # compute admittance matrix coefficients
            g = tx_calc.calculate_conductance(branch) # r / (r^2 + x^2) from m-file
            b = tx_calc.calculate_susceptance(branch) # -x / (r^2 + x^2) from m-file
            bc = branch['charging_susceptance']
            tau = 1.0 # if ratio == 0. then tau == 1. else tau == ratio (from m-file)
            shift = 0.0 # angle in radian (from m-file)
            if branch['branch_type'] == 'transformer':
                tau = branch['transformer_tap_ratio']
                shift = radians(branch['transformer_phase_shift'])
            g11 = g / tau ** 2
            g12 = g * np.cos(shift) / tau
            g21 = g * np.sin(shift) / tau
            g22 = g
            b11 = (b + bc / 2) / tau ** 2
            b12 = b * np.cos(shift) / tau
            b21 = b * np.sin(shift) / tau
            b22 = b + bc / 2
            # pf
            K_pf[wcolf, br] += g11
            K_pf[ccol, br] += (-g12+b21)
            K_pf[scol, br] += (-g21-b12)
            # pt
            K_pt[wcolt, br] += g22
            K_pt[ccol, br] += (-g12-b21)
            K_pt[scol, br] += (-g21+b12)
            # qf
            K_qf[wcolf, br] += -b11
            K_qf[ccol, br] += (b12+g21)
            K_qf[scol, br] += (b21-g12)
            # qt
            K_qt[wcolt, br] += -b22
            K_qt[ccol, br] += (b12-g21)
            K_qt[scol, br] += (b21+g12)
        self.model_data['K_pf'] = K_pf
        self.model_data['K_pt'] = K_pt
        self.model_data['K_qf'] = K_qf
        self.model_data['K_qt'] = K_qt
    
        # kcl matrices
        # real: [pg, pl, pr, pf, pt, wi, wo] * K_p = v_p
        # reac: [qg, ql, qf, qt, wi, wo] * K_q = v_q
        # TODO: use sparse matrix structures
        cols_p = np.concatenate([pgcols, plcols, prcols, pfcols, ptcols, wicols, wocols])
        cols_q = np.concatenate([qgcols, qlcols, qfcols, qtcols, wicols, wocols])
        K_p = np.zeros((len(cols_p), len(wicols)+len(wocols)), dtype=np.float32)
        K_q = np.zeros((len(cols_q), len(wicols)+len(wocols)), dtype=np.float32)

        buses = dict(self.md.elements(element_type='bus'))
        shunts = dict(self.md.elements(element_type='shunt'))
        bs, gs = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)
        for b, bdict in buses.items():
            # K_p
            pgcol = [i for i,col in enumerate(cols_p) if col.startswith('pg_b_%s_g_'%(b))]
            plcol = [i for i,col in enumerate(cols_p) if col=='pl_b_%s'%(b)]
            pfcol = [i for i,col in enumerate(cols_p) if col.startswith('pf_f_%s_t_'%(b))]
            ptcol = [i for i,col in enumerate(cols_p) if col.startswith('pt_f_') and
                     col.endswith('_t_%s'%(b))]
            pcol = np.concatenate([plcol, pfcol, ptcol]).astype(int)
            K_p[pgcol, int(b)-1] = K_p[pgcol, int(b)-1] - 1
            K_p[pcol, int(b)-1] = K_p[pcol, int(b)-1] + 1        
            if gs[b] != 0.0:
                wcol = np.where(cols_p=='w_b_%s'%(b))[0][0]
                K_p[wcol, int(b)-1] += gs[b]/mscale

            # K_q
            qgcol = [i for i,col in enumerate(cols_q) if col.startswith('qg_b_%s_g_'%(b))]
            qlcol = [i for i,col in enumerate(cols_q) if col=='ql_b_%s'%(b)]
            qfcol = [i for i,col in enumerate(cols_q) if col.startswith('qf_f_%s_t_'%(b))]
            qtcol = [i for i,col in enumerate(cols_q) if col.startswith('qt_f_') and
                     col.endswith('_t_%s'%(b))]
            qcol = np.concatenate([qlcol, qfcol, qtcol]).astype(int)
            K_q[qgcol, int(b)-1] = K_q[qgcol, int(b)-1] - 1
            K_q[qcol, int(b)-1] = K_q[qcol, int(b)-1] + 1
            if bs[b] != 0.0:
                wcol = np.where(cols_q=='w_b_%s'%(b))[0][0]
                K_q[wcol, int(b)-1] -= bs[b]/mscale
        self.model_data['K_p'] = K_p
        self.model_data['K_q'] = K_q
  
        # KVL constraints
        # get indices, signs, and cycle ids for each cycle
        cyinds = []
        cysigns = []
        cyrows = []
        for cnum, angles in self.cycles.items():
            signs = []
            inds = []
            for i, a in enumerate(angles):
                bf = a.split('_')[2]
                bt = a.split('_')[4]
                try:
                    inds.append(np.where(np.array(ccols)=='c_f_%s_t_%s'%(bf,bt))[0][0])
                    signs.append(1. if a.startswith('+') else -1.)
                except:
                    inds.append(np.where(np.array(ccols)=='c_f_%s_t_%s'%(bt,bf))[0][0])
                    signs.append(1. if a.startswith('-') else -1.)
            cyinds.append(inds)
            cysigns.append(signs)
            cyrows.append(np.repeat(cnum, len(cyinds[-1])))            
        cyinds = np.concatenate(cyinds).astype(np.int32)
        cysigns = np.concatenate(cysigns).astype(np.float32)
        cyrows = np.concatenate(cyrows).astype(np.int32)
        self.model_data['cyinds'] = cyinds
        self.model_data['cysigns'] = cysigns
        self.model_data['cyrows'] = cyrows
            
        # pythagorean constraint indices
        # used to reorder [wi, wo] tensor for c**2+s**2=w_f*w_t check
        num_cs = len(ccols)
        py_from_ind = np.zeros(num_cs)
        py_to_ind = np.zeros(num_cs)
        cols_w = np.concatenate([wicols, wocols])
        for i, col in enumerate(ccols):
            bf = col.split('_')[2]
            bt = col.split('_')[4]
            py_from_ind[i] = np.where(cols_w=='w_b_%s'%(bf))[0][0]
            py_to_ind[i] = np.where(cols_w=='w_b_%s'%(bt))[0][0]
        self.model_data['py_from_ind'] = py_from_ind.astype(int)
        self.model_data['py_to_ind'] = py_to_ind.astype(int)
            
        # pfind to align [pg, pr] and [qg] matrices for PF calculations
        pcols = np.concatenate([pgcols, prcols])
        pcols = [c[2:] for c in pcols]
        qcols = [c[2:] for c in qgcols]
        pfind = np.array([qcols.index(c) for c in pcols])
        self.model_data['pfind'] = pfind.astype(int)

    def build_test_constraints(self):
        '''
        Function to build test constraints based on parsed data
        **Note**:  do not use to train real models (test data leak)
        '''
        pg, pl, ql, wi = self.inputs
        pr, qg, wo, c, s = self.outputs
    
        constraints = {}
    
        # kcl, kvl, py constraints are data-based
        constraints['kcl'] = {'eta':0}
        constraints['kvl'] = {'eta':0}
        constraints['py'] = {'eta':0}
    
        # glim: U and L
        v = np.concatenate([pr, qg], axis=1)
        U = np.max(v, axis=0)
        L = np.min(v, axis=0)
        constraints['glim'] = {'U':U, 'L':L, 'eta':0}
    
        # vlim: U and L
        U = np.max(wo, axis=0)
        L = np.min(wo, axis=0)
        constraints['vlim'] = {'U':U, 'L':L, 'eta':0}
    
        # valim: U and L
        v = tf.math.angle(tf.complex(tf.convert_to_tensor(c), tf.convert_to_tensor(s))).numpy()
        U = np.max(v, axis=0)
        L = np.min(v, axis=0)
        constraints['valim'] = {'U':U, 'L':L, 'eta':0}
    
        # tlim: U_f and U_t
        # need to calculate flows
        z_f = np.concatenate([wi, wo, c, s], axis=1)
        pf = np.matmul(z_f, self.model_data['K_pf'])
        pt = np.matmul(z_f, self.model_data['K_pt'])
        qf = np.matmul(z_f, self.model_data['K_qf'])
        qt = np.matmul(z_f, self.model_data['K_qt'])
        v_f = pf**2+qf**2
        v_t = pt**2+qt**2
        U_f = np.max(v_f, axis=0)
        U_t = np.max(v_t, axis=0)
        constraints['tlim'] = {'U_f':U_f, 'U_t':U_t, 'eta':0}
    
        # pflim: U and L
        pgt = np.concatenate([pg, pr], axis=1)
        qgt = qg[:,self.model_data['pfind']]
        v = tf.math.angle(tf.complex(tf.convert_to_tensor(pgt), tf.convert_to_tensor(qgt))).numpy()
        U = np.max(v, axis=0)
        L = np.min(v, axis=0)
        constraints['pflim'] = {'U':U, 'L':L, 'eta':0}
    
        self.constraints = constraints
            
    def verify(self, split=None):
        '''
        Function to verify data and parsed model_data structures
        '''
        # read data using indices
        if not split:
            inputs = np.concatenate(self.inputs, axis=1)
            outputs = np.concatenate(self.outputs, axis=1) # don't need b_flags here
        elif split=='train':
            inputs, outputs = next(iter(self.train))
            inputs, outputs = inputs.numpy(), outputs.numpy()
        elif split=='val':
            inputs, outputs = next(iter(self.val))
            inputs, outputs = inputs.numpy(), outputs.numpy()
        elif split=='test':
            inputs, outputs = next(iter(self.test))
            inputs, outputs = inputs.numpy(), outputs.numpy()
        pg = inputs[:,self.iinds[0]]
        pl = inputs[:,self.iinds[1]]
        ql = inputs[:,self.iinds[2]]
        wi = inputs[:,self.iinds[3]]
        pr = outputs[:,self.oinds[0]]
        qg = outputs[:,self.oinds[1]]
        wo = outputs[:,self.oinds[2]]
        c = outputs[:,self.oinds[3]]
        s = outputs[:,self.oinds[4]]
    
        z_f = np.concatenate([wi, wo, c, s], axis=1)
        v_pf = np.matmul(z_f, self.model_data['K_pf'])
        v_pt = np.matmul(z_f, self.model_data['K_pt'])
        v_qf = np.matmul(z_f, self.model_data['K_qf'])
        v_qt = np.matmul(z_f, self.model_data['K_qt'])
        
        # kcl
        if 'kcl' in self.constraints:
            v_p = np.matmul(np.concatenate([pg, pl, pr, v_pf, v_pt, wi, wo], axis=1), self.model_data['K_p'])
            v_q = np.matmul(np.concatenate([qg, ql, v_qf, v_qt, wi, wo], axis=1), self.model_data['K_q'])
            v_kcl = np.max(np.abs(np.concatenate([v_p, v_q], axis=1)))
            print('maximum kcl violation: %.4g'%v_kcl)
        
        # kvl - mirror tensorflow ops
        if 'kvl' in self.constraints:
            ct = tf.gather(tf.convert_to_tensor(c), self.model_data['cyinds'], axis=1)
            st = self.model_data['cysigns']*tf.gather(tf.convert_to_tensor(s), self.model_data['cyinds'], axis=1)
            rc = tf.RaggedTensor.from_value_rowids(tf.transpose(ct), self.model_data['cyrows'])
            rs = tf.RaggedTensor.from_value_rowids(tf.transpose(st), self.model_data['cyrows'])
            v_kvl = tf.transpose(tf.reduce_sum(tf.math.angle(tf.complex(rc, rs)), axis=1))
            v_kvl = np.max(np.abs(v_kvl.numpy()))
            print('maximum kvl violation: %.4g'%v_kvl)
 
        # py
        if 'py' in self.constraints:
            v = np.concatenate([wi, wo], axis=1)
            v_f = np.expand_dims(v[:, self.model_data['py_from_ind']], 2)
            v_t = np.expand_dims(v[:, self.model_data['py_to_ind']], 2)
            v_ft = np.prod(np.concatenate([v_f, v_t], axis=2), axis=2)
            v_py = np.max(np.abs(c**2+s**2-v_ft))
            print('maximum py violation: %.4g'%v_py)
        
        # glim
        if 'glim' in self.constraints:
            v = np.concatenate([pr, qg], axis=1)
            v_l = self.constraints['glim']['L']-v
            v_l[v_l<0] = 0
            v_u = v-self.constraints['glim']['U']
            v_u[v_u<0] = 0
            v_glim = np.max(v_l+v_u)
            print('maximum glim violation: %.4g'%v_glim)
        
        # vlim
        if 'vlim' in self.constraints:
            v_l = self.constraints['vlim']['L']-wo
            v_l[v_l<0] = 0
            v_u = wo-self.constraints['vlim']['U']
            v_u[v_u<0] = 0
            v_vlim = np.max(v_l+v_u)
            print('maximum vlim violation: %.4g'%v_vlim)
        
        # valim - mirror tensorflow ops
        if 'valim' in self.constraints:
            v = tf.math.angle(tf.complex(tf.convert_to_tensor(c), tf.convert_to_tensor(s))).numpy()
            v_l = self.constraints['valim']['L']-v
            v_l[v_l<0] = 0
            v_u = v-self.constraints['valim']['U']
            v_u[v_u<0] = 0
            v_valim = np.max(v_l+v_u)
            print('maximum valim violation: %.4g'%v_valim)
        
        # tlim
        if 'tlim' in self.constraints:
            v_f = v_pf**2+v_qf**2-self.constraints['tlim']['U_f']
            v_f[v_f<0] = 0
            v_t = v_pt**2+v_qt**2-self.constraints['tlim']['U_t']
            v_t[v_t<0] = 0
            v_tlim = np.max(v_f+v_t)
            print('maximum tlim violation: %.4g'%v_tlim)
        
        # pflim - mirror tensorflow ops
        if 'pflim' in self.constraints:     
            pgt = tf.convert_to_tensor(np.concatenate([pg, pr], axis=1))
            qgt = tf.gather(tf.convert_to_tensor(qg), self.model_data['pfind'], axis=1)
            v = tf.math.angle(tf.complex(tf.convert_to_tensor(pgt), tf.convert_to_tensor(qgt))).numpy()
            v_l = self.constraints['pflim']['L']-v
            v_l[v_l<0] = 0
            v_u = v-self.constraints['pflim']['U']
            v_u[v_u<0] = 0
            v_pflim = np.max(v_l+v_u)
            print('maximum pflim violation: %.4g'%v_pflim)
            
    def get_splits(self, f_train=.75, f_val=.15, shuff=True, batch=256, drop_remainder=False):
        '''
        Function to generate train, validate, and test splits of the data and return
        as tf.data.Dataset objects
        '''
        b = self.b_flags
        pg, pl, ql, wi = self.inputs
        pr, qg, wo, c, s = self.outputs

        N = len(pg)
        N_train, N_val = int(f_train*N), int(f_val*N)
        N_test = N - N_train - N_val

        if shuff:
            b, pg, pl, ql, wi, pr, qg, wo, c, s = shuffle(b, pg, pl, ql, wi, pr, qg, wo, c, s)
        
        b_train, b_val, b_test = b[:N_train,:], b[N_train:-N_test,:], b[-N_test:,:]
        pg_train, pg_val, pg_test = pg[:N_train,:], pg[N_train:-N_test,:], pg[-N_test:,:]
        pl_train, pl_val, pl_test = pl[:N_train,:], pl[N_train:-N_test,:], pl[-N_test:,:]
        ql_train, ql_val, ql_test = ql[:N_train,:], ql[N_train:-N_test,:], ql[-N_test:,:]
        wi_train, wi_val, wi_test = wi[:N_train,:], wi[N_train:-N_test,:], wi[-N_test:,:]
        pr_train, pr_val, pr_test = pr[:N_train,:], pr[N_train:-N_test,:], pr[-N_test:,:]
        qg_train, qg_val, qg_test = qg[:N_train,:], qg[N_train:-N_test,:], qg[-N_test:,:]
        wo_train, wo_val, wo_test = wo[:N_train,:], wo[N_train:-N_test,:], wo[-N_test:,:]
        c_train, c_val, c_test = c[:N_train,:], c[N_train:-N_test,:], c[-N_test:,:]
        s_train, s_val, s_test = s[:N_train,:], s[N_train:-N_test,:], s[-N_test:,:]
        
        self.train = Dataset.from_tensor_slices((np.concatenate([pg_train, pl_train, 
                                                                 ql_train, wi_train], axis=1), 
                                                 np.concatenate([pr_train, qg_train, wo_train, 
                                                                 c_train, s_train, b_train], axis=1)))
        self.train = self.train.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        self.val = Dataset.from_tensor_slices((np.concatenate([pg_val, pl_val, 
                                                               ql_val, wi_val], axis=1), 
                                               np.concatenate([pr_val, qg_val, wo_val, 
                                                               c_val, s_val, b_val], axis=1)))
        self.val = self.val.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        self.test = Dataset.from_tensor_slices((np.concatenate([pg_test, pl_test, 
                                                                ql_test, wi_test], axis=1), 
                                                np.concatenate([pr_test, qg_test, 
                                                                wo_test, c_test, s_test, b_test], axis=1)))
        self.test = self.test.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        
    def save_splits(self, folder, name_prefix):
        # confirm directory exists
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        x_tr, y_tr = ([], [])
        x_va, y_va = ([], [])
        x_te, y_te = ([], [])
        
        for x, y in self.train.as_numpy_iterator():
            x_tr.append(x)
            y_tr.append(y)
        x_tr, y_tr = (np.concatenate(x_tr, axis=0), np.concatenate(y_tr, axis=0))
        
        for x, y in self.val.as_numpy_iterator():
            x_va.append(x)
            y_va.append(y)
        x_va, y_va = (np.concatenate(x_va, axis=0), np.concatenate(y_va, axis=0))

        for x, y in self.test.as_numpy_iterator():
            x_te.append(x)
            y_te.append(y)
        x_te, y_te = (np.concatenate(x_te, axis=0), np.concatenate(y_te, axis=0))
        
        splits = {'x_tr':x_tr, 'y_tr':y_tr, 'x_va':x_va, 'y_va':y_va, 'x_te':x_te, 'y_te':y_te}
        np.save(os.path.join(folder, '%s_splits.npy'%name_prefix), splits)
        
    def load_splits(self, folder, name_prefix, batch=256, drop_remainder=False):
        splits = np.load(os.path.join(folder, '%s_splits.npy'%name_prefix), allow_pickle=True).item()
        self.train = Dataset.from_tensor_slices((splits['x_tr'], splits['y_tr']))
        self.train = self.train.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        self.val = Dataset.from_tensor_slices((splits['x_va'], splits['y_va']))
        self.val = self.val.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        self.test = Dataset.from_tensor_slices((splits['x_te'], splits['y_te']))
        self.test = self.test.batch(batch, drop_remainder=drop_remainder).cache().prefetch(
            tf.data.experimental.AUTOTUNE)