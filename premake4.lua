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
       			trigger     = 'NO_QT',
                description = 'Compiles without Qt5'
    		}

            newoption {
       			trigger     = 'NO_TBB',
                description = 'Compiles without TBB'
    	    }

  			if _OPTIONS['NO_QT'] then
     			print(string.char(27) .. '[32m Building without Qt' .. string.char(27) .. '[0m')
     		else
     			with_qt = true
                defines {"QT", "QT_NO_KEYWORDS"}
  			end

            if _OPTIONS['NO_TBB'] then
                print(string.char(27) .. '[32m Building without TBB' .. string.char(27) .. '[0m')
            else
                with_tbb = true
                defines {"TBB"}
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
                end
              	includedirs {'/usr/local/include'}
            	libdirs {'/usr/local/lib'}
                buildoptions {'-std=c++17'}
               	linkoptions {'-std=c++17'}

            -- Linux specific settings
            configuration 'linux'
                links {'pthread'}

    end
