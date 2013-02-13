# Set the task name
TASK = psmc_check

# Versions
VERSION = `python psmc_check.py --version`

# Uncomment the correct choice indicating either SKA or TST flight environment
FLIGHT_ENV = SKA

BIN = psmc_check_xija
SHARE = psmc_check.py VERSION psmc_model_spec.json \
	index_template.rst psmc_check.css html4css1.css

include /proj/sot/ska/include/Makefile.FLIGHT

.PHONY: dist install

# Make a versioned distribution.  Could also use an EXCLUDE_MANIFEST
dist: version
	mkdir $(TASK)-$(VER)
	tar --exclude CVS --exclude "*~" --create --files-from=MANIFEST --file - \
	 | (tar --extract --directory $(TASK)-$(VER) --file - )
	tar --create --verbose --gzip --file $(TASK)-$(VER).tar.gz $(TASK)-$(VER)
	rm -rf $(TASK)-$(VER)

install:
#  Uncomment the lines which apply for this task
	mkdir -p $(INSTALL_BIN)
	mkdir -p $(INSTALL_SHARE)

	rsync --times --cvs-exclude $(BIN) $(INSTALL_BIN)/
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
