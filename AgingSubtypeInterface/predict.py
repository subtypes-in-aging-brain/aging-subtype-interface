import numpy as np
import cloudpickle as cp 
from urllib.request import urlopen 
import pandas as pd

def aging_subtype(Features, IntracranialCavity):
    loaded_pickle_object = cp.load(urlopen("https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/TrainedSustainModel.pickle")) 
    X = loaded_pickle_object[0]
    subtype_model = loaded_pickle_object[1]
    samples_sequence = X[0]
    samples_f = X[1]

    for i in range(len(Features)):
        Features[i]=Features[i]/IntracranialCavity 
    Features=np.asarray(Features)

    Z = pd.read_csv(urlopen("https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/Z_score_transformation_means_sds.csv"))

    SuStaInLabels = ['Brainvolume_cor','Ventricles_cor','Hippocampus_cor','precuneus_volumes_cor',
                'middletemporal_volume_cor','fusiform_volume_cor','entorhinal_volume_cor',
                'frontal_cor', 'parietal_cor','temporal_cor','occipital_cor']
    
    for i in range(len(Features)):
        idx=Z['Unnamed: 0']==SuStaInLabels[i]
        mu=Z.loc[idx,'mean']
        sigma=Z.loc[idx,'sd']
        Features[i] = (Features[i] - mu)/sigma

    Dval = np.zeros((2,len(Features)))
    for i in range(2)
        Dval[i,:] = Features
    Xnew = predict.subtype_and_stage_individuals(subtype_model,Dval,samples_sequence, samples_f, len(Features))
    return Xnew

def subtype_and_stage_individuals(subtype_model, sustainData, samples_sequence, samples_f, N_samples):
        # Subtype and stage a set of subjects. Useful for subtyping/staging subjects that were not used to build the model

        nSamples                            = sustainData.shape[0]  #data_local.shape[0]
        nStages                             = samples_sequence.shape[1]    #self.stage_zscore.shape[1]

        n_iterations_MCMC                   = samples_sequence.shape[2]
        select_samples                      = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples))
        N_S                                 = samples_sequence.shape[0]
        temp_mean_f                         = np.mean(samples_f, axis=1)
        ix                                  = np.argsort(temp_mean_f)[::-1]

        prob_subtype_stage                  = np.zeros((nSamples, nStages + 1, N_S))
        prob_subtype                        = np.zeros((nSamples, N_S))
        prob_stage                          = np.zeros((nSamples, nStages + 1))

        for i in range(N_samples):
            sample                          = int(select_samples[i])

            this_S                          = samples_sequence[ix, :, sample]
            this_f                          = samples_f[ix, sample]

            _,                  \
            _,                  \
            total_prob_stage,   \
            total_prob_subtype, \
            total_prob_subtype_stage        = calculate_likelihood(subtype_model, sustainData, this_S, this_f, nStages)

            total_prob_subtype              = total_prob_subtype.reshape(len(total_prob_subtype), N_S)
            total_prob_subtype_norm         = total_prob_subtype        / np.tile(np.sum(total_prob_subtype, 1).reshape(len(total_prob_subtype), 1),        (1, N_S))
            total_prob_stage_norm           = total_prob_stage          / np.tile(np.sum(total_prob_stage, 1).reshape(len(total_prob_stage), 1),          (1, nStages + 1)) #removed total_prob_subtype

            #total_prob_subtype_stage_norm   = total_prob_subtype_stage  / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1), 1).reshape(nSamples, 1, 1),   (1, nStages + 1, N_S))
            total_prob_subtype_stage_norm   = total_prob_subtype_stage / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1, keepdims=True), 2).reshape(nSamples, 1, 1),(1, nStages + 1, N_S))

            prob_subtype_stage              = (i / (i + 1.) * prob_subtype_stage)   + (1. / (i + 1.) * total_prob_subtype_stage_norm)
            prob_subtype                    = (i / (i + 1.) * prob_subtype)         + (1. / (i + 1.) * total_prob_subtype_norm)
            prob_stage                      = (i / (i + 1.) * prob_stage)           + (1. / (i + 1.) * total_prob_stage_norm)

        ml_subtype                          = np.nan * np.ones((nSamples, 1))
        prob_ml_subtype                     = np.nan * np.ones((nSamples, 1))
        ml_stage                            = np.nan * np.ones((nSamples, 1))
        prob_ml_stage                       = np.nan * np.ones((nSamples, 1))

        for i in range(nSamples):
            this_prob_subtype               = np.squeeze(prob_subtype[i, :])

            if (np.sum(np.isnan(this_prob_subtype)) == 0):
                this_subtype                = np.where(this_prob_subtype == np.max(this_prob_subtype))

                try:
                    ml_subtype[i]           = this_subtype
                except:
                    ml_subtype[i]           = this_subtype[0][0]
                if this_prob_subtype.size == 1 and this_prob_subtype == 1:
                    prob_ml_subtype[i]      = 1
                else:
                    try:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype]
                    except:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype[0][0]]

            this_prob_stage                 = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])
            if (np.sum(np.isnan(this_prob_stage)) == 0):
                this_stage                  = np.where(this_prob_stage == np.max(this_prob_stage))
                ml_stage[i]                 = this_stage[0][0]
                prob_ml_stage[i]            = this_prob_stage[this_stage[0][0]]

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

def calculate_likelihood(subtype_model, sustainData, S, f, nStages):
        # Computes the likelihood of a mixture of models
        #
        #
        # OUTPUTS:
        # loglike               - the log-likelihood of the current model
        # total_prob_subj       - the total probability of the current SuStaIn model for each subject
        # total_prob_stage      - the total probability of each stage in the current SuStaIn model
        # total_prob_cluster    - the total probability of each subtype in the current SuStaIn model
        # p_perm_k              - the probability of each subjects data at each stage of each subtype in the current SuStaIn model

        M                                   = sustainData.shape[0]  #data_local.shape[0]
        N_S                                 = S.shape[0]
        N                                   = nStages #sustainData.getNumStages()    #self.stage_zscore.shape[1]

        f                                   = np.array(f).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = calculate_likelihood_stage(subtype_model, sustainData, S[s])  #self.__calculate_likelihood_stage_linearzscoremodel_approx(data_local, S[s])


        total_prob_cluster                  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        loglike                             = np.sum(np.log(total_prob_subj + 1e-250))

        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

def calculate_likelihood_stage(subtype_model, sustainData, S):
        '''
         Computes the likelihood of a single linear z-score model using an
         approximation method (faster)
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        N                                   = subtype_model.stage_biomarker_index.shape[1]
        S_inv                               = np.array([0] * N)
        S_inv[S.astype(int)]                = np.arange(N)
        possible_biomarkers                 = np.unique(subtype_model.stage_biomarker_index)
        B                                   = len(possible_biomarkers)
        point_value                         = np.zeros((B, N + 2))

        # all the arange you'll need below
        arange_N                            = np.arange(N + 2)

        for i in range(B):
            b                               = possible_biomarkers[i]
            event_location                  = np.concatenate([[0], S_inv[(subtype_model.stage_biomarker_index == b)[0]], [N]])
            event_value                     = np.concatenate([[subtype_model.min_biomarker_zscore[i]], subtype_model.stage_zscore[subtype_model.stage_biomarker_index == b], [subtype_model.max_biomarker_zscore[i]]])
            for j in range(len(event_location) - 1):

                if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits

                    # original
                    #temp                   = np.arange(event_location[j],event_location[j+1]+2)
                    #point_value[i,temp]    = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)

                    # fastest by a bit
                    temp                    = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp]    = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

                else:
                    # original
                    #temp                   = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    #point_value[i, temp]   = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

                    # fastest by a bit
                    temp                    = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp]    = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

        stage_value                         = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

        M                                   = sustainData.shape[0]   #data_local.shape[0]
        p_perm_k                            = np.zeros((M, N + 1))

        # optimised likelihood calc - take log and only call np.exp once after loop
        sigmat = np.array(subtype_model.std_biomarker_zscore)

        factor                              = np.log(1. / np.sqrt(np.pi * 2.0) * sigmat)
        coeff                               = np.log(1. / float(N + 1))

        # original
        """
        for j in range(N+1):
            x                   = (data-np.tile(stage_value[:,j],(M,1)))/sigmat
            p_perm_k[:,j]       = coeff+np.sum(factor-.5*x*x,1)
        """
        # faster - do the tiling once
        # stage_value_tiled                   = np.tile(stage_value, (M, 1))
        # N_biomarkers                        = stage_value.shape[0]
        # for j in range(N + 1):
        #     stage_value_tiled_j             = stage_value_tiled[:, j].reshape(M, N_biomarkers)
        #     x                               = (sustainData.data - stage_value_tiled_j) / sigmat  #(data_local - stage_value_tiled_j) / sigmat
        #     p_perm_k[:, j]                  = coeff + np.sum(factor - .5 * np.square(x), 1)
        # p_perm_k                            = np.exp(p_perm_k)

        # even faster - do in one go
        x = (sustainData[:, :, None] - stage_value) / sigmat[None, :, None]
        p_perm_k = coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1)
        p_perm_k = np.exp(p_perm_k)

        return p_perm_k

def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N