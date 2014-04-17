"""
Voxel-Based Morphometry on Oasis dataset.
Relationship between aging and gray matter density.

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

n_subjects = 50

### Get data
# DARTEL data
dataset_files_dartel = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
subject_ids_dartel = dataset_files_dartel.ext_vars['id']
# non-DARTEL data
dataset_files_nondartel = datasets.fetch_oasis_vbm(n_subjects=n_subjects,
                                                   dartel_version=False)
subject_ids_nondartel = dataset_files_nondartel.ext_vars['id']
# consider only common subjects
subject_ids_common = np.intersect1d(subject_ids_dartel, subject_ids_nondartel)
subject_ids_mask_dartel = np.zeros(dataset_files_dartel.ext_vars.size,
                                   dtype=bool)
gm_maps_dartel = []
for i, subject_id in enumerate(dataset_files_dartel.ext_vars['id']):
    if subject_id in subject_ids_common:
        gm_maps_dartel.append(dataset_files_dartel.gray_matter_maps[i])
        subject_ids_mask_dartel[i] = True
ext_vars = dataset_files_dartel.ext_vars[subject_ids_mask_dartel]
gm_maps_nondartel = []
for i, subject_id in enumerate(dataset_files_nondartel.ext_vars['id']):
    if subject_id in subject_ids_common:
        gm_maps_nondartel.append(dataset_files_nondartel.gray_matter_maps[i])
# get externals vars
age = ext_vars['age'].astype(float).reshape((-1, 1))

# Prepare plots
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")
picked_slice = 36
vmin = -np.log10(0.1)  # 10% corrected
vmax = 3.
for i, (gm_maps, title) in enumerate(zip([gm_maps_nondartel, gm_maps_dartel],
                                         ["non-DARTEL", "DARTEL"])):
    ### Mask data
    nifti_masker = NiftiMasker(
        memory='nilearn_cache',
        memory_level=1)  # cache options
    # remove features with too low between-subject variance
    gm_maps_masked = nifti_masker.fit_transform(gm_maps)
    gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
    # final masking
    new_images = nifti_masker.inverse_transform(gm_maps_masked)
    gm_maps_masked = nifti_masker.fit_transform(new_images)
    n_samples, n_features = gm_maps_masked.shape
    print "DARTEL:", n_samples, "subjects, ", n_features, "features"

    ### Perform massively univariate analysis with permuted OLS ###############
    print "Massively univariate model"
    neg_log_pvals, all_scores, _ = permuted_ols(
        age, gm_maps_masked,  # + intercept as a covariate by default
        n_perm=1000,
        n_jobs=-1)  # can be changed to use more CPUs
    neg_log_pvals_unmasked = nifti_masker.inverse_transform(
        neg_log_pvals).get_data()[..., 0]

    ### Show results
    print "Plotting results"
    # background anat
    mean_anat = nibabel.load(gm_maps[0]).get_data()
    for img in gm_maps[1:]:
        mean_anat += nibabel.load(img).get_data()
    mean_anat /= float(len(gm_maps))
    masked_pvals = np.ma.masked_less(neg_log_pvals_unmasked, vmin)
    ax = grid[i]
    ax.imshow(np.rot90(mean_anat[..., picked_slice]),
               interpolation='nearest', cmap=plt.cm.gray, vmin=0., vmax=1.)
    im = ax.imshow(np.rot90(masked_pvals[..., picked_slice]),
                   interpolation='nearest', cmap=plt.cm.autumn,
                   vmin=vmin, vmax=vmax)
    ax.set_title(r'Negative $\log_{10}$ p-values'
                 + '\n(%s data)\n%d detections'
                 % (title, (masked_pvals[..., picked_slice] > vmin).sum()))
    ax.axis('off')

grid[0].cax.colorbar(im)
plt.subplots_adjust(0.05, 0., .9, 0.95)

plt.show()
