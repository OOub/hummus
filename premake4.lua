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

            newoption {
       			trigger     = 'no_qt',
                description = 'Compiles without Qt'
    		}

            newoption {
       			trigger     = 'no_tbb',
                description = 'Compiles without tbb'
    	    }

  			if _OPTIONS['no_qt'] then
     			print(string.char(27) .. '[32m Building without Qt' .. string.char(27) .. '[0m')
     		else
     			with_qt = true
  			end

            if _OPTIONS['no_tbb'] then
                print(string.char(27) .. '[32m Building without TBB' .. string.char(27) .. '[0m')
            else
                with_tbb = true
            end

      			-- All files in source, third_party and applications
          	files {'source/**.hpp',
                   'third_party/**.hpp',
          		   'applications/' .. name .. '.cpp'
          	}

      		if with_qt then
				-- Qt-dependent files
				files(qt.moc({'source/GUI/input_viewer.hpp',
							  'source/GUI/output_viewer.hpp',
					           'source/GUI/dynamics_viewer.hpp'
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
                if with_tbb then
                    links {'tbb'}
                    defines {"TBB"}
                end
              	includedirs {'/usr/local/include'}
            	libdirs {'/usr/local/lib'}
                buildoptions {'-std=c++17'}
               	linkoptions {'-std=c++17'}

            -- Linux specific settings
            configuration 'linux'
                links {'pthread'}

    end
