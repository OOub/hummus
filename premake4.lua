local qt = require 'qt'

solution 'hummus' 
    configurations {'Release', 'Debug'}
    location 'build'

    for index, file in pairs(os.matchfiles('applications/*.cpp')) do
    	local name = path.getbasename(file)
    	project(name)
    		-- General settings
    		kind 'ConsoleApp'
    		language 'C++'
        	location 'build'

        	-- Build Options
        	newoption {
   				trigger     = 'no-qt',
   				description = 'Compiles without Qt'
			}

			if _OPTIONS['no-qt'] then
   				print(string.char(27) .. '[32m Building without Qt' .. string.char(27) .. '[0m')
   			else
   				with_qt = true
			end

			-- All files in source
        	files {'source/**.hpp',
        		'source/addons/**.hpp', 
        		'source/dependencies/**.hpp', 
        		'source/GUI/qt/**.hpp',
        		'sourbrewce/GUI/puffin/**.hpp', 
        		'source/learningRules/**.hpp', 
        		'source/neurons/**.hpp', 
        		'source/synapticKernels/**.hpp', 
        		'source/networkExtensions/**.hpp', 
        		'source/randomDistributions/**.hpp', 
        		'applications/' .. name .. '.cpp'
        	}

			if with_qt then
				-- Qt-dependent files
				files(qt.moc({'source/GUI/qt/inputViewer.hpp', 
							  'source/GUI/qt/outputViewer.hpp', 
					          'source/GUI/qt/potentialViewer.hpp'
					          }, 
							  'build/moc'))

	            includedirs(qt.includedirs())
	            libdirs(qt.libdirs())
	            links(qt.links())
	            buildoptions(qt.buildoptions())
	            linkoptions(qt.linkoptions())
	        end

	        -- Declare the configurations
	        configuration 'Release'
	            targetdir 'build/release'
	            defines {'NDEBUG'}
	            flags {'OptimizeSpeed'}

	        configuration 'Debug'
	            targetdir 'build/debug'
	            defines {'DEBUG'}
	            flags {'Symbols'}

	        configuration 'linux or macosx'
            	includedirs {'/usr/local/include'}
	        	libdirs {'/usr/local/lib'}
  				defines { "UNIX" }

	        -- Linux specific settings
	        configuration 'linux'
	        	links {'pthread', 'sqlite3', 'tbb'}
	            buildoptions {'-std=c++11'}
	           	linkoptions {'-std=c++11'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	        	links {'sqlite3', 'tbb'}
	            buildoptions {'-std=c++11'}
	           	linkoptions {'-std=c++11'}
end
