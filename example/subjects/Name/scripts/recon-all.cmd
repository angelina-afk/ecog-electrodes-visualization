

#---------------------------------
# New invocation of recon-all Tue Mar 12 21:10:52 UTC 2019 

 mri_convert /tmp/subjects/mri_yacyuk/IM-0001-0002.dcm /tmp/subjects/Yacyuk/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Tue Mar 12 21:11:19 UTC 2019

 cp /tmp/subjects/Yacyuk/mri/orig/001.mgz /tmp/subjects/Yacyuk/mri/rawavg.mgz 


 mri_convert /tmp/subjects/Yacyuk/mri/rawavg.mgz /tmp/subjects/Yacyuk/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /tmp/subjects/Yacyuk/mri/transforms/talairach.xfm /tmp/subjects/Yacyuk/mri/orig.mgz /tmp/subjects/Yacyuk/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Tue Mar 12 21:11:38 UTC 2019

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

#--------------------------------------------
#@# Talairach Failure Detection Tue Mar 12 21:13:07 UTC 2019

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /usr/local/freesurfer/bin/extract_talairach_avi_QA.awk /tmp/subjects/Yacyuk/mri/transforms/talairach_avi.log 


 tal_QC_AZS /tmp/subjects/Yacyuk/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Tue Mar 12 21:13:07 UTC 2019

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 


 mri_add_xform_to_header -c /tmp/subjects/Yacyuk/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Tue Mar 12 21:15:04 UTC 2019

 mri_normalize -g 1 -mprage nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Tue Mar 12 21:17:13 UTC 2019

 mri_em_register -rusage /tmp/subjects/Yacyuk/touch/rusage.mri_em_register.skull.dat -skull nu.mgz /usr/local/freesurfer/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta 


 mri_watershed -rusage /tmp/subjects/Yacyuk/touch/rusage.mri_watershed.dat -T1 -brain_atlas /usr/local/freesurfer/average/RB_all_withskull_2016-05-10.vc700.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


 cp brainmask.auto.mgz brainmask.mgz 

#-------------------------------------
#@# EM Registration Tue Mar 12 21:36:34 UTC 2019

 mri_em_register -rusage /tmp/subjects/Yacyuk/touch/rusage.mri_em_register.dat -uns 3 -mask brainmask.mgz nu.mgz /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Tue Mar 12 21:51:02 UTC 2019

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Tue Mar 12 21:52:37 UTC 2019

 mri_ca_register -rusage /tmp/subjects/Yacyuk/touch/rusage.mri_ca_register.dat -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Tue Mar 12 23:53:01 UTC 2019

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca aseg.auto_noCCseg.mgz 


 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /tmp/subjects/Yacyuk/mri/transforms/cc_up.lta Yacyuk 

#--------------------------------------
#@# Merge ASeg Wed Mar 13 00:43:23 UTC 2019

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Wed Mar 13 00:43:23 UTC 2019

 mri_normalize -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Wed Mar 13 00:46:18 UTC 2019

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Wed Mar 13 00:46:19 UTC 2019

 mri_segment -mprage brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Wed Mar 13 00:48:21 UTC 2019

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.auto_noCCseg.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Wed Mar 13 00:48:56 UTC 2019

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Wed Mar 13 00:49:00 UTC 2019

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Wed Mar 13 00:49:05 UTC 2019

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Wed Mar 13 00:49:09 UTC 2019

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Wed Mar 13 00:49:13 UTC 2019

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Wed Mar 13 00:49:39 UTC 2019

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Wed Mar 13 00:50:06 UTC 2019

 mris_sphere -q -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Wed Mar 13 00:52:57 UTC 2019

 mris_sphere -q -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#--------------------------------------------
#@# Fix Topology Copy lh Wed Mar 13 00:56:00 UTC 2019

 cp ../surf/lh.orig.nofix ../surf/lh.orig 


 cp ../surf/lh.inflated.nofix ../surf/lh.inflated 

#--------------------------------------------
#@# Fix Topology Copy rh Wed Mar 13 00:56:00 UTC 2019

 cp ../surf/rh.orig.nofix ../surf/rh.orig 


 cp ../surf/rh.inflated.nofix ../surf/rh.inflated 

#@# Fix Topology lh Wed Mar 13 00:56:00 UTC 2019

 mris_fix_topology -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_fix_topology.lh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 Yacyuk lh 

#@# Fix Topology rh Wed Mar 13 01:10:55 UTC 2019

 mris_fix_topology -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_fix_topology.rh.dat -mgz -sphere qsphere.nofix -ga -seed 1234 Yacyuk rh 


 mris_euler_number ../surf/lh.orig 


 mris_euler_number ../surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm ../surf/rh.inflated 

#--------------------------------------------
#@# Make White Surf lh Wed Mar 13 01:25:32 UTC 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs Yacyuk lh 

#--------------------------------------------
#@# Make White Surf rh Wed Mar 13 01:29:40 UTC 2019

 mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs Yacyuk rh 

#--------------------------------------------
#@# Smooth2 lh Wed Mar 13 01:33:39 UTC 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Wed Mar 13 01:33:43 UTC 2019

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Wed Mar 13 01:33:48 UTC 2019

 mris_inflate -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_inflate.lh.dat ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Wed Mar 13 01:34:13 UTC 2019

 mris_inflate -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_inflate.rh.dat ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Wed Mar 13 01:34:40 UTC 2019

 mris_curvature -w lh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Wed Mar 13 01:35:45 UTC 2019

 mris_curvature -w rh.white.preaparc 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 


#-----------------------------------------
#@# Curvature Stats lh Wed Mar 13 01:36:53 UTC 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm Yacyuk lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Wed Mar 13 01:36:56 UTC 2019

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm Yacyuk rh curv sulc 

#--------------------------------------------
#@# Sphere lh Wed Mar 13 01:36:59 UTC 2019

 mris_sphere -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_sphere.lh.dat -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Wed Mar 13 01:57:43 UTC 2019

 mris_sphere -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_sphere.rh.dat -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Wed Mar 13 02:26:09 UTC 2019

 mris_register -curv -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_register.lh.dat ../surf/lh.sphere /usr/local/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Wed Mar 13 03:01:36 UTC 2019

 mris_register -curv -rusage /tmp/subjects/Yacyuk/touch/rusage.mris_register.rh.dat ../surf/rh.sphere /usr/local/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Wed Mar 13 03:40:54 UTC 2019

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Wed Mar 13 03:40:56 UTC 2019

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Wed Mar 13 03:40:57 UTC 2019

 mrisp_paint -a 5 /usr/local/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Wed Mar 13 03:40:58 UTC 2019

 mrisp_paint -a 5 /usr/local/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Wed Mar 13 03:41:00 UTC 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk lh ../surf/lh.sphere.reg /usr/local/freesurfer/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Wed Mar 13 03:41:10 UTC 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk rh ../surf/rh.sphere.reg /usr/local/freesurfer/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# Make Pial Surf lh Wed Mar 13 03:41:20 UTC 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs Yacyuk lh 

#--------------------------------------------
#@# Make Pial Surf rh Wed Mar 13 03:53:12 UTC 2019

 mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg ../mri/aseg.presurf -mgz -T1 brain.finalsurfs Yacyuk rh 

#--------------------------------------------
#@# Surf Volume lh Wed Mar 13 04:04:45 UTC 2019
#--------------------------------------------
#@# Surf Volume rh Wed Mar 13 04:04:47 UTC 2019
#--------------------------------------------
#@# Cortical ribbon mask Wed Mar 13 04:04:50 UTC 2019

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon Yacyuk 

#-----------------------------------------
#@# Parcellation Stats lh Wed Mar 13 04:11:01 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab Yacyuk lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab Yacyuk lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Wed Mar 13 04:11:46 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab Yacyuk rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab Yacyuk rh pial 

#-----------------------------------------
#@# Cortical Parc 2 lh Wed Mar 13 04:12:32 UTC 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk lh ../surf/lh.sphere.reg /usr/local/freesurfer/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Wed Mar 13 04:12:46 UTC 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk rh ../surf/rh.sphere.reg /usr/local/freesurfer/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Parcellation Stats 2 lh Wed Mar 13 04:13:00 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab Yacyuk lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Wed Mar 13 04:13:23 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab Yacyuk rh white 

#-----------------------------------------
#@# Cortical Parc 3 lh Wed Mar 13 04:13:47 UTC 2019

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk lh ../surf/lh.sphere.reg /usr/local/freesurfer/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Wed Mar 13 04:13:57 UTC 2019

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 Yacyuk rh ../surf/rh.sphere.reg /usr/local/freesurfer/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Parcellation Stats 3 lh Wed Mar 13 04:14:08 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab Yacyuk lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Wed Mar 13 04:14:31 UTC 2019

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab Yacyuk rh white 

#-----------------------------------------
#@# WM/GM Contrast lh Wed Mar 13 04:14:54 UTC 2019

 pctsurfcon --s Yacyuk --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Wed Mar 13 04:15:02 UTC 2019

 pctsurfcon --s Yacyuk --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Wed Mar 13 04:15:10 UTC 2019

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# AParc-to-ASeg aparc Wed Mar 13 04:15:29 UTC 2019

 mri_aparc2aseg --s Yacyuk --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt 

#-----------------------------------------
#@# AParc-to-ASeg a2009s Wed Mar 13 04:19:24 UTC 2019

 mri_aparc2aseg --s Yacyuk --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --a2009s 

#-----------------------------------------
#@# AParc-to-ASeg DKTatlas Wed Mar 13 04:23:20 UTC 2019

 mri_aparc2aseg --s Yacyuk --volmask --aseg aseg.presurf.hypos --relabel mri/norm.mgz mri/transforms/talairach.m3z /usr/local/freesurfer/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt --annot aparc.DKTatlas --o mri/aparc.DKTatlas+aseg.mgz 

#-----------------------------------------
#@# APas-to-ASeg Wed Mar 13 04:27:16 UTC 2019

 apas2aseg --i aparc+aseg.mgz --o aseg.mgz 

#--------------------------------------------
#@# ASeg Stats Wed Mar 13 04:27:22 UTC 2019

 mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /usr/local/freesurfer/ASegStatsLUT.txt --subject Yacyuk 

#-----------------------------------------
#@# WMParc Wed Mar 13 04:29:10 UTC 2019

 mri_aparc2aseg --s Yacyuk --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc.mgz --ctxseg aparc+aseg.mgz 


 mri_segstats --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject Yacyuk --surf-wm-vol --ctab /usr/local/freesurfer/WMParcStatsLUT.txt --etiv 

#--------------------------------------------
#@# BA_exvivo Labels lh Wed Mar 13 04:37:28 UTC 2019

 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA1_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA2_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA3a_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA3b_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA4a_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA4p_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA6_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA44_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA45_exvivo.label --trgsubject Yacyuk --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.V1_exvivo.label --trgsubject Yacyuk --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.V2_exvivo.label --trgsubject Yacyuk --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.MT_exvivo.label --trgsubject Yacyuk --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject Yacyuk --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject Yacyuk --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s Yacyuk --hemi lh --ctab /usr/local/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.entorhinal_exvivo.label --l lh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s Yacyuk --hemi lh --ctab /usr/local/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab Yacyuk lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab Yacyuk lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Wed Mar 13 04:41:09 UTC 2019

 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA1_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA2_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA3a_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA3b_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA4a_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA4p_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA6_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA44_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA45_exvivo.label --trgsubject Yacyuk --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.V1_exvivo.label --trgsubject Yacyuk --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.V2_exvivo.label --trgsubject Yacyuk --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.MT_exvivo.label --trgsubject Yacyuk --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject Yacyuk --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject Yacyuk --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /tmp/subjects/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject Yacyuk --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s Yacyuk --hemi rh --ctab /usr/local/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.entorhinal_exvivo.label --l rh.perirhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s Yacyuk --hemi rh --ctab /usr/local/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab Yacyuk rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab Yacyuk rh white 

