solution 'icarus'
    configurations {'Release', 'Debug'}
    location 'build'

    newaction {
        trigger = "install",
        description = "Install the library",
        execute = function ()
            os.execute('rm -rf build')
            os.execute('mkdir build')
            os.execute('cp -r source/. build')
            print(string.char(27) .. '[32mcpp_snn library installed.' .. string.char(27) .. '[0m')
            os.exit()
        end
    }

    newaction {
        trigger = 'uninstall',
        description = 'Remove all the files installed during build processes',
        execute = function ()
            os.execute('rm -rf build')
            print(string.char(27) .. '[32mcpp_snn library uninstalled.' .. string.char(27) .. '[0m')
            os.exit()
        end
    }

    project 'unsupervisedNetwork'
        -- General settings
        kind 'ConsoleApp'
        language 'C++'
        location 'build'
        files {'source/**.hpp', 'applications/unsupervisedNetwork.cpp'}

		-- Run moc and link to the Qt library
        local mocFiles = {
            'source/inputViewer.hpp',
            'source/outputViewer.hpp',
            'source/potentialViewer.hpp',
        }

        -- Linux specific settings
        configuration 'linux'
        if os.is("linux") then
            local mocCommand = '/home/omar/Qt/5.9.1/gcc_64/bin/moc' -- must point to the moc executable
            local qtIncludeDirectory = '/home/omar/Qt/5.9.1/gcc_64/include' -- Qt headers
            local qtLibDirectory = '/home/omar/Qt/5.9.1/gcc_64/lib' -- Qt dynamic libraries
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
        configuration 'macosx'
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

	project 'testNetwork'
	        -- General settings
	        kind 'ConsoleApp'
	        language 'C++'
	        location 'build'
	        files {'source/**.hpp', 'applications/testNetwork.cpp'}

			-- Run moc and link to the Qt library
	        local mocFiles = {
	            'source/inputViewer.hpp',
	            'source/outputViewer.hpp',
	            'source/potentialViewer.hpp',
	        }

	        -- Linux specific settings
	        configuration 'linux'
	        if os.is("linux") then
	            local mocCommand = '/home/omar/Qt/5.9.1/gcc_64/bin/moc' -- must point to the moc executable
	            local qtIncludeDirectory = '/home/omar/Qt/5.9.1/gcc_64/include' -- Qt headers
	            local qtLibDirectory = '/home/omar/Qt/5.9.1/gcc_64/lib' -- Qt dynamic libraries
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
	        configuration 'macosx'
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

	project 'supervisedNetwork'
	        -- General settings
	        kind 'ConsoleApp'
	        language 'C++'
	        location 'build'
	        files {'source/**.hpp', 'applications/supervisedNetwork.cpp'}

			-- Run moc and link to the Qt library
	        local mocFiles = {
	            'source/inputViewer.hpp',
	            'source/outputViewer.hpp',
	            'source/potentialViewer.hpp',
	        }

	        -- Linux specific settings
	        configuration 'linux'
	        if os.is("linux") then
	            local mocCommand = '/home/omar/Qt/5.9.1/gcc_64/bin/moc' -- must point to the moc executable
	            local qtIncludeDirectory = '/home/omar/Qt/5.9.1/gcc_64/include' -- Qt headers
	            local qtLibDirectory = '/home/omar/Qt/5.9.1/gcc_64/lib' -- Qt dynamic libraries
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
	        configuration 'macosx'
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

