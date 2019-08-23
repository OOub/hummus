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

			-- All files in source, third_party and applications
        	files {'source/**.hpp',
        		'source/addons/**.hpp', 
        		'source/GUI/**.hpp',
        		'source/learningRules/**.hpp', 
        		'source/neurons/**.hpp', 
        		'source/synapticKernels/**.hpp', 
        		'source/randomDistributions/**.hpp',
                'third_party/**.hpp',
        		'applications/' .. name .. '.cpp'
        	}

			if with_qt then
				-- Qt-dependent files
				files(qt.moc({'source/GUI/inputViewer.hpp', 
							  'source/GUI/outputViewer.hpp', 
					          'source/GUI/dynamicsViewer.hpp'
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
	            flags {'OptimizeSpeed','FloatFast'}

	        configuration 'Debug'
	            targetdir 'build/debug'
	            defines {'DEBUG'}
	            flags {'Symbols'}

	        configuration 'linux or macosx'
            	includedirs {'/usr/local/include'}
	        	libdirs {'/usr/local/lib'}

	        -- Linux specific settings
	        configuration 'linux'
                if with_tbb then
	        	    links {'pthread'}
                else
                    links {'pthread'}
                end
	            buildoptions {'-std=c++11'}
	           	linkoptions {'-std=c++11'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++11'}
	           	linkoptions {'-std=c++11'}
end
