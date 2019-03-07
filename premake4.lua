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
   				trigger     = 'without-qt',
   				description = 'Compiles without Qt'
			}

			newoption {
   				trigger     = 'tbb',
   				description = 'Compiles with Intel TBB'
			}

			if _OPTIONS['without-qt'] then
   				print(string.char(27) .. '[32m Building without Qt' .. string.char(27) .. '[0m')
   			else
   				with_qt = true
			end

			if _OPTIONS['tbb'] then
				defines { }
			end

			-- All files in source
        	files {'source/**.hpp',
        		'source/addOns/**.hpp', 
        		'source/dependencies/**.hpp', 
        		'source/GUI/**.hpp', 
        		'source/learningRules/**.hpp', 
        		'source/neurons/**.hpp', 
        		'source/synapticKernels/**.hpp', 
        		'source/networkExtensions/**.hpp', 
        		'applications/' .. name .. '.cpp'
        	}

			if with_qt then
				-- Qt-dependent files
				files(qt.moc({'source/GUI/inputViewer.hpp', 
							  'source/GUI/outputViewer.hpp', 
					          'source/GUI/potentialViewer.hpp'
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
  				defines { "POSIX" }
  				linkoptions {'-ltbb'}

	        -- Linux specific settings
	        configuration 'linux'
	        	links {'pthread'}
	            buildoptions {'-std=c++11'}
	            linkoptions {'-std=c++11'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++11'}
                linkoptions {'-std=c++11'}

            -- Windows specific settings
            configuration 'windows'
            	files {'.clang-format'}
            	defines { "WINDOWS" }
end
