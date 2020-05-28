# makefile for RT physics modules, called from main makefile
# must create libphy_modules.a in its root directory

$(info -------------------------------------------------------------- )
$(info Modules Alfrodull Makefile )

# set some variables if not set
includedir ?= unset
h5include ?= unset
cpp_flags ?= unset
cuda_flags ?= unset
arch ?= unset
CC_comp_flag ?= unset
MODE ?= unset

$(info Some variables inherited from parent makefiles)
$(info includes: $(includedir))
$(info h5includes: $(h5include))
$(info cpp_flags: $(cpp_flags))
$(info cuda_flags: $(cuda_flags))
$(info CC compile flag: $(CC_comp_flag))
$(info dependeny flags: $(dependency_flags))
$(info arch: $(arch))
$(info MODE: $(MODE))

######################################################################
# Directories
THOR_ROOT = ../

# Includes
LOCAL_INCLUDE = src/inc 
LOCAL_INCLUDE_PHY = thor_module/inc

# shared modules
SHARED_MODULES_INCLUDE = $(THOR_ROOT)src/physics/modules/inc/

# thor root include if we want to use code from there
THOR_INCLUDE = $(THOR_ROOT)src/headers

# source dirs
LOCAL_SOURCES = src
SHARED_MODULES_DIR = $(THOR_ROOT)src/physics/modules/src/
# object directory
BUILDDIR = obj
OUTPUTDIR = $(MODE)

######################################################################
$(info Sub Makefile variables)
$(info THOR root from submakefile: $(THOR_ROOT))

######################################################################
all: libphy_modules.a

# path to local module code
vpath %.cu src src/kernels src/opacities thor_module/src
vpath %.cpp $(LOCAL_SOURCES) 
vpath %.h $(LOCAL_INCLUDE) $(LOCAL_INCLUDE_PHY) 
# path to thor headers
vpath %.h $(THOR_INCLUDE)
# path to phy_modules
vpath %.h $(SHARED_MODULES_INCLUDE)
vpath %.cu $(SHARED_MODULES_DIR)
vpath %.cpp $(SHARED_MODULES_DIR)



# objects
obj_cuda   := alfrodull_engine.o alfrodullib.o planck_table.o opacities.o atomic_add.o calculate_physics.o correct_surface_emission.o integrate_flux.o interpolate_values.o math_helpers.o insolation_angle.o two_streams_radiative_transfer.o      
obj_cpp := gauss_legendre_weights.o 
obj := $(obj_cpp) $(obj_cuda)


ifndef VERBOSE
.SILENT:
endif

#######################################################################
# create directory

$(BUILDDIR):
	mkdir $@

$(BUILDDIR)/${OUTPUTDIR}: $(BUILDDIR)
		mkdir -p $(BUILDDIR)/$(OUTPUTDIR)

#######################################################################
# build objects

INCLUDE_DIRS = -I$(SHARED_MODULES_INCLUDE) -I$(THOR_INCLUDE) -I$(LOCAL_INCLUDE) -I$(LOCAL_INCLUDE_PHY) 


#######################################################################
# target to reuse radiative transfer.

$(BUILDDIR)/${OUTPUTDIR}/radiative_transfer.d: $(SHARED_MODULES_DIR)radiative_transfer.cu | $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(BLUE)computing dependencies $@ $(END)
	set -e; rm -f $@; \
	$(CC) $(dependencies_flags) $(arch) $(cuda_dep_flags) $(h5include) $(INCLUDE_DIRS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(BUILDDIR)/$(OUTPUTDIR)/radiative_transfer.o: $(SHARED_MODULES_DIR)radiative_transfer.cu $(BUILDDIR)/$(OUTPUTDIR)/radiative_transfer.d| $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(YELLOW)creating $@ $(END)
	if test $$CDB = "-MJ" ; then \
		$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(h5libdir) $(INCLUDE_DIRS) $(CDB) $@.json -o $@ $<; \
	else \
		$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(h5libdir) $(INCLUDE_DIRS) -o $@ $<; \
	fi

#######################################################################
# make dependency targets
# this generates obj/(debug|release)/*.d files containing dependencies targets.
# it uses the compiler to find all the header files a CUDA/C++ file depends on
# those files are then included at the end
# make is run twice, once to create the dependency targets and once to run the compilation

# for CUDA files
$(BUILDDIR)/${OUTPUTDIR}/%.d: %.cu | $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(BLUE)computing dependencies $@ $(END)
	set -e; rm -f $@; \
	$(CC) $(dependencies_flags) $(arch) $(cuda_dep_flags) $(h5include) $(INCLUDE_DIRS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$


# for C++ files
$(BUILDDIR)/${OUTPUTDIR}/%.d: %.cpp | $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(BLUE)computing dependencies $@ $(END)
	set -e; rm -f $@; \
	$(CC) $(dependencies_flags) $(arch) $(cpp_dep_flags) $(h5include) $(INCLUDE_DIRS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

#######################################################################
# build objects
# CUDA files
$(BUILDDIR)/$(OUTPUTDIR)/%.o: %.cu $(BUILDDIR)/$(OUTPUTDIR)/%.d| $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(YELLOW)creating $@ $(END)
	if test $$CDB = "-MJ" ; then \
		$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(h5libdir) $(INCLUDE_DIRS) $(CDB) $@.json -o $@ $<; \
	else \
		$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(h5libdir) $(INCLUDE_DIRS) -o $@ $<; \
	fi

# C++ files
$(BUILDDIR)/$(OUTPUTDIR)/%.o: %.cpp $(BUILDDIR)/$(OUTPUTDIR)/%.d| $(BUILDDIR)/$(OUTPUTDIR) $(BUILDDIR)
	@echo $(YELLOW)creating $@ $(END)
	if test $$CDB = "-MJ" ; then \
		$(CC) $(CC_comp_flag) $(arch) $(cpp_flags) $(h5include) -I$(includedir) $(CDB) $(INCLUDE_DIRS)  $@.json -o $@ $<; \
	else \
		$(CC) $(CC_comp_flag) $(arch) $(cpp_flags) $(h5include) -I$(includedir) $(INCLUDE_DIRS) -o $@ $<; \
	fi


libphy_modules.a: $(addprefix $(BUILDDIR)/$(OUTPUTDIR)/,$(obj)) $(BUILDDIR)/${OUTPUTDIR}/radiative_transfer.o $(BUILDDIR)/${OUTPUTDIR}/phy_modules.o | $(BUILDDIR)
	@echo $(YELLOW)creating $@ $(END)
	@echo $(GREEN)Linking Modules into static lib $(END)
	ar rcs $@ $(BUILDDIR)/${OUTPUTDIR}/phy_modules.o $(addprefix $(BUILDDIR)/$(OUTPUTDIR)/,$(obj)) $(BUILDDIR)/${OUTPUTDIR}/radiative_transfer.o

#######################################################################
# Cleanup
.phony: clean,ar
clean:
	@echo $(CYAN)clean up library $(END)
	-$(RM) libphy_modules.a
	@echo $(CYAN)clean up modules objects $(END)
	-$(RM) $(BUILDDIR)/debug/*.o
	-$(RM) $(BUILDDIR)/debug/*.o.json
	-$(RM) $(BUILDDIR)/release/*.o
	-$(RM) $(BUILDDIR)/release/*.o.json
	-$(RM) $(BUILDDIR)/prof/*.o
	-$(RM) $(BUILDDIR)/prof/*.o.json
	@echo $(CYAN)clean up dependencies $(END)
	-$(RM) $(BUILDDIR)/debug/*.d
	-$(RM) $(BUILDDIR)/debug/*.d.*
	-$(RM) $(BUILDDIR)/release/*.d
	-$(RM) $(BUILDDIR)/release/*.d.*
	-$(RM) $(BUILDDIR)/prof/*.d
	-$(RM) $(BUILDDIR)/prof/*.d.*
	-$(RM) -d $(BUILDDIR)/debug/
	-$(RM) -d $(BUILDDIR)/release/
	-$(RM) -d $(BUILDDIR)/prof/
	@echo $(CYAN)remove modules object dir $(END)
	-$(RM) -d $(BUILDDIR)
$(info -------------------------------------------------------------- )
