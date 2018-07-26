
solution 'adonis_t' 
    configurations {'Release', 'Debug'}
    location 'build'

    os.execute('rm -rf build') 
    os.execute('mkdir build')

    for index, file in pairs(os.matchfiles('applications/*.cpp')) do
    	local name = path.getbasename(file)
    	project(name)
    		-- General settings
    		kind 'ConsoleApp'
    		language 'C++'
        	location 'build'
        	files {'source/**.hpp', 'applications/' .. name .. '.cpp'}

        	newoption {
   				trigger     = 'without-qt',
   				description = 'Compiles without Qt'
			}

			if _OPTIONS['without-qt'] then
   				print(string.char(27) .. '[32m Building without Qt' .. string.char(27) .. '[0m')
   			else
   				with_qt = true
   				print(string.char(27) .. '[32m Building with Qt' .. string.char(27) .. '[0m')
			end

			if with_qt then
				print(string.char(27) .. '[32m entering loop' .. string.char(27) .. '[0m')
	            -- Run moc and link to the Qt library
	        	local mocFiles = {
	            	'source/inputViewer.hpp',
	            	'source/outputViewer.hpp',
	            	'source/potentialViewer.hpp',
	        	}

		        -- Linux specific settings
		        if os.is("linux") then
		            local mocCommand = '/home/omar/Qt/5.10.1/gcc_64/bin/moc' -- must point to the moc executable
		            local qtIncludeDirectory = '/home/omar/Qt/5.10.1/gcc_64/include' -- Qt headers
		            local qtLibDirectory = '/home/omar/Qt/5.10.1/gcc_64/lib' -- Qt dynamic libraries
		            local mocDirectory = path.getdirectory(_SCRIPT) .. '/build/moc'
		            os.rmdir(mocDirectory)
		            os.mkdir(mocDirectory)
		            for index, mocFile in pairs(mocFiles) do
		                if os.execute(mocCommand
		                    .. ' -I\'' .. qtIncludeDirectory .. '/QtQml\''
		                    .. ' -o \'' .. mocDirectory .. '/' .. path.getbasename(mocFile) .. '.cpp\''
		                    .. ' \''.. mocFile .. '\''
		                    ) ~= 0 then
		                    print(string.char(27) .. '[31mPre-compiling ' .. mocFile .. ' failed' .. string.char(27) .. '[0m')
		                    os.exit(1)
		                end
		            print(string.char(27) .. '[32m' .. mocFile .. ' was successfully pre-compiled' .. string.char(27) .. '[0m')
		            end

		            files {mocDirectory .. '/**.hpp', mocDirectory .. '/**.cpp', 'source/**.qml'}
		            includedirs {qtIncludeDirectory, qtIncludeDirectory .. '/QtQml'}
		            
		            libdirs {qtLibDirectory}
		            links {'Qt5Core', 'Qt5Gui', 'Qt5Qml', 'Qt5Quick','Qt5Widgets','Qt5Charts'}
		            
		            buildoptions {'-fPIC'}
		        end


		        -- Mac OS X specific settings
		        if os.is("macosx") then

		            local mocCommand = '/usr/local/opt/qt/bin/moc' -- must point to the moc executable
		            local qtIncludeDirectory = '/usr/local/opt/qt/include' -- Qt headers
		            local qtLibDirectory = '/usr/local/opt/qt/lib' -- Qt dynamic libraries

		            local mocDirectory = path.getdirectory(_SCRIPT) .. '/build/moc'
		            os.rmdir(mocDirectory)
		            os.mkdir(mocDirectory)
		            for index, mocFile in pairs(mocFiles) do
		                if os.execute(mocCommand
		                    .. ' -I\'' .. qtIncludeDirectory .. '/QtQml\''
		                    .. ' -o \'' .. mocDirectory .. '/' .. path.getbasename(mocFile) .. '.cpp\''
		                    .. ' \''.. mocFile .. '\''
		                    ) ~= 0 then
		                    print(string.char(27) .. '[31mPre-compiling ' .. mocFile .. ' failed' .. string.char(27) .. '[0m')
		                    os.exit(1)
		                end
		                print(string.char(27) .. '[32m' .. mocFile .. ' was successfully pre-compiled' .. string.char(27) .. '[0m')
		            end

		            files {mocDirectory .. '/**.hpp', mocDirectory .. '/**.cpp', 'source/**.qml'}
		            includedirs {qtIncludeDirectory, qtIncludeDirectory .. '/QtQml'}

		            libdirs {qtLibDirectory}
		            buildoptions {'-fPIC'}

		            linkoptions
		            {
		                '-F' .. qtLibDirectory,
		                '-framework QtCore',
		                '-framework QtGui',
		                '-framework QtQml',
		                '-framework QtQuick',
		                '-framework QtWidgets',
		                '-framework QtQuickControls2',
		                '-framework QtCharts',
		            }
		        end
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

	        -- Linux specific settings
	        configuration 'linux'
	            buildoptions {'-std=c++11'}
	            linkoptions {'-std=c++11'}
	            links {'pthread'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++11', '-stdlib=libc++'}
	            linkoptions {'-std=c++11','-stdlib=libc++'}

	        -- Configuration for both
	        configuration {}
	            includedirs {'/usr/local/include'}
	            libdirs {'/usr/local/lib'}
end