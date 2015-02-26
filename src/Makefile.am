
SUBDIRS =

noinst_PROGRAMS = dune_mlmc

dune_mlmc_SOURCES = dune_mlmc.cc

dune_mlmc_CPPFLAGS = $(AM_CPPFLAGS) \
	$(DUNEMPICPPFLAGS) \
	$(UG_CPPFLAGS) \
	$(ALBERTA_CPPFLAGS) \
	$(ALUGRID_CPPFLAGS)
# The libraries have to be given in reverse order (most basic libraries
# last).  Also, due to some misunderstanding, a lot of libraries include the
# -L option in LDFLAGS instead of LIBS -- so we have to include the LDFLAGS
# here as well.
dune_mlmc_LDADD = \
	$(DUNE_LDFLAGS) $(DUNE_LIBS) \
	$(ALUGRID_LDFLAGS) $(ALUGRID_LIBS) \
	$(ALBERTA_LDFLAGS) $(ALBERTA_LIBS) \
	$(UG_LDFLAGS) $(UG_LIBS) \
	$(DUNEMPILIBS)	\
	$(LDADD)
dune_mlmc_LDFLAGS = $(AM_LDFLAGS) \
	$(DUNEMPILDFLAGS) \
	$(UG_LDFLAGS) \
	$(ALBERTA_LDFLAGS) \
	$(ALUGRID_LDFLAGS) \
	$(DUNE_LDFLAGS)

# don't follow the full GNU-standard
# we need automake 1.9
AUTOMAKE_OPTIONS = foreign 1.9

# pass most important options when "make distcheck" is used
DISTCHECK_CONFIGURE_FLAGS = --with-dune-multiscale=$(DUNE_MULTISCALE_ROOT)  CXX="$(CXX)" CC="$(CC)"

EXTRA_DIST = CMakeLists.txt

include $(top_srcdir)/am/global-rules

