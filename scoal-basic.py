import numpy as np
from scipy import *
from scipy import sparse, linalg
import logging, time

from scikits.learn import linear_model

#Define :P
ax = {0:"row", 1:"col"}
ROW=0
COL=1

# You can add any model to this dictionary that had the following calls
# model.fit(NxM sparse matrix)
# model.predict(NxP) -> P-values
reg_models = {"linear":linear_model.LinearRegression,
              "linear-sgd":linear_model.SGDRegressor}

reg_args   = {"linear":(),
              "linear-sgd":()}

reg_kwargs = {"linear":{"fit_intercept":False},
              "linear-sgd":{"alpha":0.001, "n_iter":1}}

def add_regressor(name, f, args=(), kwargs={}):
    global reg_kwargs, reg_models, reg_args
    reg_models[name] = f
    reg_args[name] = args
    reg_kwargs[name] = kwargs

class ScoalRegressor(object):
    def __init__(self, M, row_clust_num, col_clust_num, debug_level=logging.DEBUG):
        # Setup logging 
        self.log = logging.getLogger("scoal")
        self.log.setLevel(logging.DEBUG)
        # Store matrix
        self.M = sparse.coo_matrix(M)
        self.log.info("Loaded %s (%s) matrix." % (str(self.M), str(self.M.shape)))

        # Record the row and col cluster counts
        self.rowClusterCount = row_clust_num
        self.colClusterCount = col_clust_num
        self.indxClusterCount = (row_clust_num, col_clust_num)

        # These values can be "take"en from the global array indices 
        self.modCols   = {}
        self.modRows   = {}

        # This stores the various models, and thier sse's
        self.models    = {}
        self.model_sse = {}

        # Setup the row and column assignments, and the initial models        
        self.rowAttrs = []
        self.colAttrs = []

        self.rowAttrsDescrip = []
        self.colAttrsDescrip = []

        # Make it general so axis based code can work
        self.axisAttrs = (self.rowAttrs, self.colAttrs)
        self.axisAttrsDescrip = (self.rowAttrsDescrip, self.colAttrsDescrip)

        # Matrices that store the sse error of various models keyed on (row, col, model)
        # returns array rowshape X colshape
        self.error_mats = {}
        self.current_mse = inf
        
        # Starting at zero . . .
        self.I = 0
    
    def addAxAttr(self, ax, aray, description):
        assert (aray.size == self.M.shape[ax]), "Attribute must match matrix dimensions!"
        self.axisAttrs[ax].append(aray)
        self.axisAttrsDescrip[ax].append(description)
    def addRowAttr(self, aray, description):
        self.addAxAttr(ROW, aray, description)
    def addColAttr(self, aray, description):
        self.addAxAttr(COL, aray, description)

    def initialize(self):
        # Initialize the assignments if necessary
        if not hasattr(self, "rowAssignments"):
            self.rowAssignments = random.randint(0, self.rowClusterCount, size=self.M.shape[ROW])
        if not hasattr(self, "colAssignments"):
            self.colAssignments = random.randint(0, self.colClusterCount, size=self.M.shape[COL])

        self.axisAssignments = (self.rowAssignments, self.colAssignments)

        self.computeCCMembership()
        self.computeAllModels()
        self.computeAllErrors()

    # Syntatic sugar to below function
    def reClusterRows(self, *args, **kwargs):
        return self.reClusterAxis(ROW, *args, **kwargs)
    def reClusterCols(self, *args, **kwargs):
        return self.reClusterAxis(COL, *args, **kwargs)

    def reClusterAxis(self, axis, model="linear", max_swap_fraction=1):
        new_class_bins = self.indxClusterCount[axis]
        error_accumulator = zeros((self.M.shape[axis], new_class_bins))

        for cc_data in self.getCCIter(full_only = False):
            for cc_model in self.getCCIter():
                curr_rc, curr_cc = cc_model
                ccm = (curr_rc, curr_cc, model)

                # Grab the features from a certain cocluster
                ri, ci, names, features, targets = self.getCCFeatures(*cc_data)
                # Make a prediction of all of them in a different cocluster
                predictions = self.models[ccm].predict(features)
                errors = (array(targets - predictions)**2).flatten()
                
                if axis == ROW: 
                    ii = ri
                    model_index = curr_rc
                if axis == COL: 
                    ii = ci
                    model_index = curr_cc

                if ii.size == 0:
                    continue

                # Trick stolen from Clint
                rows_to_update = bincount(ii) > 0
                row_errors = bincount(ii, errors)
                error_accumulator[rows_to_update, model_index] += row_errors[rows_to_update]

        # Naming these make the next semantics easier
        new_assign = error_accumulator.argmin(axis=1)
        old_assign = self.axisAssignments[axis].copy()

        new_error  = error_accumulator
        swaps = new_assign != self.axisAssignments[axis]

        # if we are swapping all of them . . .
        if max_swap_fraction == 1:
            self.axisAssignments[axis][:] = new_assign
            swap_count = sum(1 * swaps)
            return swap_count
        
        # swap a random fraction of the ones in question

        # Accumulate the swap count here . . .
        swap_count = 0
        for current_axis_group in range(new_class_bins):
            # This is a boolean slice that is true for swaps in this class
            potential_swaps = (old_assign != new_assign) & (old_assign == current_axis_group)

            # Count of the above, and imputed actual (floor div so models can never go below 1 :P)
            number_of_p_swaps = sum(1*potential_swaps)
            number_of_actual_swaps = floor(number_of_p_swaps * max_swap_fraction)
            swap_count += number_of_actual_swaps
            
            # Make a slice with the right number of swaps . . . then shuffle it
            will_swap = zeros(number_of_p_swaps, dtype=int8)
            will_swap[0:number_of_actual_swaps] = 1
            random.shuffle(will_swap)
            
            # And this agains potential swaps in our class, then assign the ones that pass
            potential_swaps[potential_swaps] &= will_swap
            self.axisAssignments[axis][potential_swaps] = new_assign[potential_swaps]

        return swap_count

    def deriveAxisFeatures(self, axis):
        '''This derives a mean feature on the axis.'''
        cm = self.m.copy()
        cm.data = 1
        
        axis_sum = self.m.sum(axis = 1-axis)
        axis_nnz = cm.sum(axis = 1-axis)
        
        axis_mean = (axis_sum / axis_nnz)

        self.addAxAttr(axis, axis_mean, "AUTOGEN:Mean-%i" % axis)

    def autoFeatures(self):
        '''Calls the above function for both rows and cols'''
        for x in range(2):
            self.deriveAxisFeatures(axis)
        

    def getCCDataSlice(self, row_cluster, col_cluster):
        '''This function returns the slice which when applied 
        to the M.row, m.col and m.data, returns all entries in a cocluster'''
        data_in_rc = self.rowAssignments[self.M.row] == row_cluster
        data_in_cc = self.colAssignments[self.M.col] == col_cluster
        return data_in_rc & data_in_cc

    def getCCDataNumber(self, row_cluster, col_cluster):
        '''Return the number of data points in a co-cluster'''
        return sum(1*self.getCCDataSlice(row_cluster, col_cluster))
    
    def getCCSubMatrix(self, row_cluster, col_cluster, sort=True):
        '''Return a sub-matrix of the current cocluster.  Use later . . . TODO'''
        # Get the slice that describes points in our cocluster
        data_slice = self.getCCDataSlice(row_cluster, col_cluster)

        # Retrieve the rows, cols, and datae related to that
        global_rows = self.M.row[data_slice]
        global_cols = self.M.col[data_slice]
        global_data = self.M.data[data_slice]

        # Find the unique global rows included
        urows = unique(global_rows)
        ucols = unique(global_cols)

        # Sort if desired.  
        if sort:
            urows.sort()
            ucols.sort()

        # Setup a new keying of the global rows to local ones
        nrows = arange(urows.size, dtype=uint32)
        ncols = arange(ucols.size, dtype=uint32)
        
        # Convert the global row and col numbers into the new local ones
        # TODO: Uglee and probably slow . . .  
        grow_to_lrow, gcol_to_lcol = {}, {}
        grow_to_lrow.update(zip(urows, nrows))
        gcol_to_lcol.update(zip(ucols, ncols))

        # Create a new matrix with the right size . . .
        new_matrix = sparse.lil_matrix((urows.size, ucols.size))
        # Populate it . . .
        for gr, gc, gv in zip(global_rows, global_cols, global_data):
            # Convert the global coords to local ones
            lr = grow_to_lrow[gr]
            lc = gcol_to_lcol[gc]
            new_matrix[lr, lc] = gv
        
        # Pass the local to global keying, and the new matrix
        return urows, ucols, new_matrix

    def getCCData(self, row_cluster, col_cluster):
        # Get the values in a given row/col cluster
        sl = self.getCCDataSlice(row_cluster, col_cluster)
        return M.data[sl]
    
    def computeCCMembership(self):
        '''Compute the membership of all co-clusters '''
        self.cc_membership = zeros((self.rowClusterCount, self.colClusterCount), dtype=int32)
        for cc_tuple, n in ndenumerate(self.cc_membership):
            rcn, ccn = cc_tuple
            self.cc_membership[cc_tuple] = self.getCCDataNumber(*cc_tuple)            
        print self.cc_membership

    def computeCCEntropy(self):
        '''Compute the entropy of all assignments'''
        self.computeCCMembership()
        # Compute the fraction of entries that fall in a cocluster
        cc_member_fraction = (1.*self.cc_membership)/self.cc_membership.sum()
        # Get rid of the empty coclusters
        fracs = cc_member_fraction[cc_member_fraction != 0].flatten()
        # Compute the entropy
        return -sum(fracs * log2(fracs))

    def getCCIter(self, full_only = True):
        '''Get an iterator that goes over all the non-empty co-clusters.  
        If full_only is set to false, this iterates over all co-clusters.'''
        full_pairs = []
        for cc, n in ndenumerate(self.cc_membership):
            if (self.cc_membership[cc] == 0) and full_only: 
                continue
            full_pairs.append(cc)

        return iter(full_pairs)

    def getAllFeatures(self):
        '''For every data-point in the matrix, get the row and col features.
        TODO:Put in inner loop to avoid slicing whole data each iteration.'''
        row_features = tuple([take(ra, self.M.row) for ra in self.rowAttrs])
        col_features = tuple([take(ca, self.M.col) for ca in self.colAttrs])
        mean_feature = tuple([ ones(self.M.row.size) ])
        all_features = vstack(mean_feature + row_features + col_features).T

        feature_names = (("Global Mean",)+tuple(self.rowAttrsDescrip)+tuple(self.rowAttrsDescrip))
        model_target  = self.M.data
        return feature_names, all_features, model_target

    def getCCFeatures(self, rcn, ccn):
        '''Like above, but only for a given row/col cluster'''
        cc_tuple = rcn, ccn
        sl = self.getCCDataSlice(rcn, ccn)
        
        row_indices = self.M.row[sl]
        row_features = tuple([take(ra, row_indices) for ra in self.rowAttrs])

        col_indices = self.M.col[sl]
        col_features = tuple([take(ca, col_indices) for ca in self.colAttrs])
        
        mean_feature = tuple([ ones(sum(1*sl)) ])
        all_features = vstack(mean_feature + row_features + col_features).T

        feature_names = (("Global Mean",)+tuple(self.rowAttrsDescrip)+tuple(self.rowAttrsDescrip))

        model_target  = self.M.data[sl]
        return row_indices, col_indices, feature_names, all_features, model_target

    def computeCCModel(self, rcn, ccn, model="linear"):
        '''Compute the model for a co-cluster.  The model must appear in the dictionary 
        populated in the header, and provide fit and predict functions.'''
        ccm_tuple = rcn, ccn, model
        # Get all the row-col features for a cocluster
        ri, ci, feature_names, data_to_model, model_target = self.getCCFeatures(rcn, ccn) 

        # Initiate a model on the features . . .
        model_args     =  reg_args[model]
        model_kwargs   =  reg_kwargs[model]
        model_instance = reg_models[model](*model_args, **model_kwargs)
        
        # Fit the model
        model_instance.fit(data_to_model, model_target)

        assert hasattr(model_target, "predict"), "Model %s does not present .predict() . . ."

        # Save it for later use . . .
        self.models[ccm_tuple] = model_instance

    def computeCCError(self, rcn, ccn, model="linear"):
        '''Compute the error of a co-cluster for a given model.  Store in dictionary and array form.'''
        ccm_tuple = rcn, ccn, model
        ri, ci, feature_names, data_to_model, target = self.getCCFeatures(rcn, ccn) 
        model_predictions = array(self.models[ccm_tuple].predict(data_to_model)).flatten()
        
        sse = sum( (target - model_predictions) ** 2 )
        self.model_sse[(rcn, ccn)] = sse
        self.sse_mat[rcn, ccn] = sse

        
    def computeAllModels(self):
        '''Call computeCCModel on all non-empty co-clusters.'''
        for cc in self.getCCIter():
            self.computeCCModel(*cc)

    def computeAllErrors(self, models=["linear"]):
        '''Call computeCCError on all non-empty co-clusters.'''
        self.sse_mat = zeros((self.rowClusterCount,self.colClusterCount))
        for cc in self.getCCIter():
            self.computeCCError(*cc)

        print self.sse_mat
        print "Total SSE: %f" % self.sse_mat.sum()
        print "Total MSE: %f" % (1. * self.sse_mat.sum() / self.M.data.size)
        print "Entropy: %f" % self.computeCCEntropy()

    def iterate(self):
        '''Do one assignment iteration of the Scoal Regressor'''
        # Update the row clusters - update the models
        rs = self.reClusterRows(max_swap_fraction=0.5)
        self.computeCCMembership()
        self.computeAllModels()

        # Update the col clusters - update the models
        cs = self.reClusterCols(max_swap_fraction=0.5)
        self.computeCCMembership()
        self.computeAllModels()

        # compute the errors . . . 
        self.computeAllErrors()

        # Update the iteration count
        self.I += 1

        swaps = rs + cs

    def converge(self, eps):
        '''Iterate the ScoalRegressor until a given epsilon is reached, or there are no more swaps.'''
        while True:
            last_mse = self.current_mse
            swaps = self.iterate()
            if swaps == 0: 
                break
            self.current_mse = (1. * self.sse_mat.sum() / self.M.data.size)
            if self.current_mse - last_mse < eps: break
            

def scoal_mat(m):
    sr = ScoalRegressor(m, 4, 5)

    # col_mean = m.mean(axis=ROW)
    # row_mean = m.mean(axis=COL)
    sr.addColAttr(col_mean, "Col-Mean")
    sr.addRowAttr(row_mean, "Row-Mean")
    sr.initialize()
    return sr


def loadMVLENS():
    print "Parsing 1M movielens data . . ."
    data_array = loadtxt("/home/meawoppl/datasets/movielens-1m/ratings.dat", delimiter="::")
    print "\tDone. . . "
    
    print "Converting to sparse matrix format . . ."
    users, movies, ratings, trash = data_array.T
    ratings -= ratings.mean()
    del trash

    print users.min(), movies.min()
    user_movie = c_[users, movies].T
    m = sparse.coo_matrix((ratings, user_movie))
    print "\tDone"

    tm = array(m.todense())
    tmm = 1*(tm != 0)


    # remove nonparticaptory users . . .
    tm = tm[tmm.sum(axis=1)!=0, :]
    tmm = tmm[tmm.sum(axis=1)!=0, :]
    tm = tm[:, tmm.sum(axis=0)!=0]
    tmm = tmm[:, tmm.sum(axis=0)!=0]

    row_mean = tm.sum(axis=1) / tmm.sum(axis=1)
    col_mean = tm.sum(axis=0) / tmm.sum(axis=0)

    
    m=sparse.coo_matrix(tm)

    return m, row_mean, col_mean

def loadERIM():
    f = open("/home/meawoppl/datasets/gdatasets/ERIM/Z.mtx")
    comment= f.readline()
    shape  = tuple([ int(i) for i in f.readline().strip().split() ])
    data_aray = loadtxt(f)
    data_aray = data_aray.reshape(shape)
    row_mean = data_aray.mean(axis=1)
    col_mean = data_aray.mean(axis=0)
    return data_aray, row_mean, col_mean


if __name__ == "__main__":
    #m, row_mean, col_mean = loadMVLENS()
    m, row_mean, col_mean = loadERIM()
    sr = scoal_mat(m)
    sr.converge(eps = 0.001)
