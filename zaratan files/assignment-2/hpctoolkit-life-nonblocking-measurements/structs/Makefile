MEAS_DIR =  /home/krajeshk/CMSC416/assignment-2/hpctoolkit-life-nonblocking-measurements
GPUBIN_CFG = no
CPU_ANALYZE = 1
GPU_ANALYZE = 1
PAR_SIZE = 100000000
JOBS = 64
SJOBS = 32
STHREADS = 2
LJOBS = 4
LTHREADS = 16
PROFTT = /cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/hpctoolkit-2023.03.01-aqszyzjguqujywzpzeyewxdcm4q4z2vv/libexec/hpctoolkit/hpcproftt
STRUCT= /cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/hpctoolkit-2023.03.01-aqszyzjguqujywzpzeyewxdcm4q4z2vv/bin/hpcstruct
CACHE= 

#*******************************************************************************
# a helper template makefile used by hpcstruct at runtime
#
# if hpcstruct is passed the name of a measurements directory that contains
# a gpubins subdirectory, this makefile will be used to orchestrate parallel
# analysis of all gpubins within the subdirectory.
#
# to simplify things at execution time, this makefile will be incorporated
# into hpcstruct as a string and written to a temporary file if it is needed.
# this avoids the need for hpcstruct to know how to find a copy of this
# makefile at runtime in an hpctoolkit installation.
#*******************************************************************************

#-------------------------------------------------------------------------------
# set up subdirectory paths
#-------------------------------------------------------------------------------
GPUBIN_DIR  = $(MEAS_DIR)/gpubins
STRUCTS_DIR = $(MEAS_DIR)/structs


#*******************************************************************************
# calculate the alternate kind of GPU CFG analysis
#*******************************************************************************
ifeq ($(GPUBIN_CFG),yes)
GPUBIN_CFG_ALT = no
else
GPUBIN_CFG_ALT = yes
endif


#*******************************************************************************
# use a measurement cache, if available
#*******************************************************************************
ifneq ($(CACHE)x,x)
CACHE_ARGS = -c $(CACHE)
else
CACHE_ARGS = --nocache
endif


#*******************************************************************************
# identify load modules to analyze
#*******************************************************************************
LM_DIR      = $(MEAS_DIR)/lm

#-------------------------------------------------------------------------------
# $(H): hpcrun file for all threads
#-------------------------------------------------------------------------------
H := $(wildcard $(MEAS_DIR)/*.hpcrun)

#-------------------------------------------------------------------------------
# $(L): load modules for hpcrun files
#-------------------------------------------------------------------------------
L := $(patsubst $(MEAS_DIR)/%.hpcrun,$(LM_DIR)/%.lm,$(H))

#-------------------------------------------------------------------------------
# create $(LM_DIR)/all.lm: a list of all load modules involved in the execution
#-------------------------------------------------------------------------------
$(LM_DIR)/all.lm: $(L)
	cat $(L) | sort -u | grep -v libhpcrun | grep -v libmonitor | grep -v libxed | grep -v libpfm | grep -v libcuda | grep -v libcupti > $(LM_DIR)/all.lm


#*******************************************************************************
# enable analysis of GPU binaries
#*******************************************************************************
ifeq ($(GPU_ANALYZE),1)
GPUBIN_USED_DIR  = $(MEAS_DIR)/gpubins-used

#-------------------------------------------------------------------------------
# create gpubins-used directory containing links to all GPU binaries used
#-------------------------------------------------------------------------------
$(GPUBIN_USED_DIR): $(LM_DIR)/all.lm
	-@mkdir $(GPUBIN_USED_DIR) >&- 2>&-
	-@cd $(GPUBIN_USED_DIR) >&- 2>&-; for i in `cat $(LM_DIR)/all.lm | grep gpubin`; do ln -s $$i; done >&- 2>&-

#-------------------------------------------------------------------------------
# $(GB): gpubin files
#-------------------------------------------------------------------------------
GB := $(wildcard $(GPUBIN_USED_DIR)/*)

#-------------------------------------------------------------------------------
# $(GS): hpcstruct files for gpubins
#-------------------------------------------------------------------------------
GS := $(patsubst $(GPUBIN_USED_DIR)/%,$(STRUCTS_DIR)/%-gpucfg-$(GPUBIN_CFG).hpcstruct,$(GB))

#-------------------------------------------------------------------------------
# $(GW): warning files that may be generated during structure analysis of gpubins
#-------------------------------------------------------------------------------
GW := $(patsubst %.hpcstruct,%.warnings,$(GS))

endif

#*******************************************************************************
#*******************************************************************************


#*******************************************************************************
# enable analysis of CPU binaries
#*******************************************************************************
ifeq ($(CPU_ANALYZE),1)
CPUBIN_DIR  = $(MEAS_DIR)/cpubins

#-------------------------------------------------------------------------------
# create cpubins directory containing links to all CPU binaries
#-------------------------------------------------------------------------------
$(CPUBIN_DIR): $(LM_DIR)/all.lm
	-@mkdir $(CPUBIN_DIR) >&- 2>&-
	-@cd $(CPUBIN_DIR) >&- 2>&-; for i in `cat $(LM_DIR)/all.lm | grep -v gpubin`; do ln -s $$i; done >&- 2>&-

#-------------------------------------------------------------------------------
# $(CB): cpubin files
#-------------------------------------------------------------------------------
CB := $(wildcard $(CPUBIN_DIR)/*)

#-------------------------------------------------------------------------------
# $(CS): hpcstruct files for cpubins
#-------------------------------------------------------------------------------
CS := $(patsubst $(CPUBIN_DIR)/%,$(STRUCTS_DIR)/%.hpcstruct,$(CB))

#-------------------------------------------------------------------------------
# $(CW): warning files that may be generated during structure analysis of cpubins
#-------------------------------------------------------------------------------
CW := $(patsubst %.hpcstruct,%.warnings,$(CS))

endif
#-------------------------------------------------------------------------------
# execute the sequence of commands for each target in a single shell
#-------------------------------------------------------------------------------
.ONESHELL:
.SILENT:

.DEFAULT_GOAL := all

#-------------------------------------------------------------------------------
# rule for analyzing a cpu binary
# 1. analyze a cpu binary file in $(CPUBIN)
# 2. produce a hpcstruct file in $(STRUCTS_DIR)
# 3. leave a warnings file in $(STRUCTS_DIR) if trouble arises
# 4. announce when analysis of a cpu binary begins and ends
#-------------------------------------------------------------------------------
$(STRUCTS_DIR)/%.hpcstruct: $(CPUBIN_DIR)/%
	@cpubin_name=`basename -s x $<`
	struct_name=$@
	warn_name=$(STRUCTS_DIR)/$$cpubin_name.warnings
	# @echo  DEBUG cpubin = $$cpu_bin_name
	nbytes=`du -b -L $< | tail -1 | awk '{ print $$1 }'`
	meas_dir=$(MEAS_DIR)
	# echo DEBUG meas_dir = $$meas_dir

	if test $$nbytes -gt $(CPAR_SIZE) ; then
		# inform the user the analysis is starting
		PARSTAT=concurrent
		if test $(THREADS) -gt 1 ; then
		    echo  \ begin parallel analysis of CPU binary $$cpubin_name \(size = $$nbytes, threads = $(THREADS)\)
		    PARSTAT=parallel
		else
		    echo \ begin concurrent analysis of CPU binary $$cpubin_name \(size = $$nbytes, threads = 1\)
		fi

		#  invoke hpcstruct on the CPU binary in the measurements directory
		$(STRUCT) $(CACHE_ARGS) -j $(THREADS) -o $$struct_name -M $$meas_dir $< > $$warn_name 2>&1
		# echo DEBUG: hpcstruct for analysis of CPU binary $$cpubin_name returned

		# See if there is anything to worry about in the warnings file
		#  suppress any ADVICE, INFO, DEBUG, and CACHESTAT lines and any blank lines;
		#  it's an error if anything remains
                #
		errs=`sed 's/^$//INFO/g;' $$warn_name |grep -v DEBUG | grep -v CACHESTAT | grep -v INFO | grep -v ADVICE | wc -l`

		# echo DEBUG errs = XX $$errs XX
		if [ $${errs} -eq 1 ] ; then
		    echo WARNING: incomplete analysis of $$cpubin_name';' see $$warn_name for details
		fi

		# extract the status relative to the cache
		CACHE_STAT=`grep CACHESTAT $$warn_name | sed 's/CACHESTAT// ' `
		# echo DEBUG CACHE_STAT = XX $$CACHE_STAT XX
		echo \ \ \ end  $$PARSTAT analysis of CPU binary $$cpubin_name $$CACHE_STAT

	fi

#-------------------------------------------------------------------------------
# rule  for analyzing a gpubin
# 1. analyze a gpubin file in $(GPUBIN_DIR)
# 2. produce a hpcstruct file in $(STRUCTS_DIR)
# 3. leave a warnings file in $(STRUCTS_DIR) if trouble arises
# 4. announce when analysis of a gpubin begins and ends
#-------------------------------------------------------------------------------
$(STRUCTS_DIR)/%-gpucfg-$(GPUBIN_CFG).hpcstruct: $(GPUBIN_DIR)/%
	@gpubin_name=`basename -s x $<`
	struct_name=$@
	rm -f $(STRUCTS_DIR)/$$gpubin_name-gpucfg-$(GPUBIN_CFG_ALT).hpcstruct
	rm -f $(STRUCTS_DIR)/$$gpubin_name-gpucfg-$(GPUBIN_CFG_ALT).warnings
	warn_name=$(STRUCTS_DIR)/$$gpubin_name-gpucfg-$(GPUBIN_CFG).warnings
	nbytes=`du -b -L $< | tail -1 | awk '{ print $$1 }'`
	meas_dir=$(MEAS_DIR)
	# echo DEBUG meas_dir = $$meas_dir

	if test $$nbytes -gt $(GPAR_SIZE) ; then
		# tell user we're starting
		PARSTAT=concurrent
		if test $(THREADS) -gt 1 ; then
		    PARSTAT=parallel
		    echo \ begin parallel [gpucfg=$(GPUBIN_CFG)] analysis of GPU binary $$gpubin_name \(size = $$nbytes, threads = $(THREADS)\)
		else
		    echo \ begin concurrent [gpucfg=$(GPUBIN_CFG)] analysis of GPU binary $$gpubin_name \(size = $$nbytes, threads = 1\)
		fi

		# invoke hpcstruct to process the gpu binary
		$(STRUCT) $(CACHE_ARGS) -j $(THREADS) --gpucfg $(GPUBIN_CFG) -o $$struct_name -M $$meas_dir $< > $$warn_name 2>&1
		# echo debug: hpcstruct for analysis of GPU binary $$gpubin_name returned

		# See if there is anything to worry about in the warnings file
		#  suppress any ADVICE, INFO, DEBUG, and CACHESTAT lines and any blank lines;
		#  it's an error if anything remains
		#
		errs=`sed 's/^$//INFO/g;' $$warn_name |grep -v DEBUG | grep -v CACHESTAT | grep -v INFO | grep -v ADVICE | wc -l`
		# echo DEBUG errs = XX $$errs XX

		if [ $${errs} -eq 1 ] ; then
		    echo WARNING: incomplete analysis of $$gpubin_name';' see $$warn_name for details
		fi

		# extract the status relative to the cache
		CACHE_STAT=`grep CACHESTAT $$warn_name  | sed 's/CACHESTAT// ' `
		# echo DEBUG CACHE_STAT = XX $$CACHE_STAT XX

		echo \ \ \ end  $$PARSTAT [gpucfg=$(GPUBIN_CFG)] analysis of GPU binary $$gpubin_name $$CACHE_STAT
	fi

#-------------------------------------------------------------------------------
# analyze files to create structure files
#-------------------------------------------------------------------------------
DOMAKE=1

ifeq ($(L),)
all:
	echo ERROR: directory $(MEAS_DIR) does not contain any hpcrun measurement files
DOMAKE=0
endif

ifeq ($(DOMAKE),1)
all: $(CPUBIN_DIR) $(GPUBIN_USED_DIR)
	$(MAKE) -j $(LJOBS) THREADS=$(LTHREADS) GPAR_SIZE=$(PAR_SIZE) CPAR_SIZE=$(PAR_SIZE) analyze
	$(MAKE) -j $(SJOBS) THREADS=$(STHREADS) GPAR_SIZE=0 CPAR_SIZE=0 analyze
endif

analyze: $(GS) $(CS)

#-------------------------------------------------------------------------------
# rule for listing load modules in hpcrun files
#-------------------------------------------------------------------------------
$(LM_DIR)/%.lm: $(MEAS_DIR)/%.hpcrun
	-@mkdir $(LM_DIR) >&- 2>&-
	$(PROFTT) -l $< > $@

#-------------------------------------------------------------------------------
# remove all generated files
#-------------------------------------------------------------------------------
clean:
	@echo removing all hpcstruct files in $(STRUCTS_DIR)
	@rm -f $(GS)
	@rm -f $(CS)
	@echo removing all links to CPU binaries in $(CPUBIN_DIR)
	@rm -rf $(CPUBIN_DIR)
	@rm -rf $(LM_DIR)
	@echo removing all warnings files in $(STRUCTS_DIR)
	@rm -f $(CW)
	@rm -f $(GW)

