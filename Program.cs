using System;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

class Program
{
    static readonly HttpClient client = new HttpClient();
    static readonly int targetPort = 8572;
    static readonly string launcherScriptName = "launch_server.py"; // Start Python script that starts the server

    static async Task Main(string[] args)
    {
        Console.WriteLine("[Main] Program started.");

        if (args.Length < 1)
        {
            Console.Error.WriteLine("[Main] No arguments provided.");
            Environment.Exit(1);
        }
        Console.WriteLine($"[Main] Arguments received: {string.Join(" ", args)}");

        // Trying to dectect the port
        Console.WriteLine($"[Main] Checking if port {targetPort} is in use...");
        if (!IsPortInUse(targetPort))
        {
            Console.WriteLine($"[Main] Port {targetPort} is not in use. Attempting to launch server.");

            // Get the path for launching the launch_server.py script
            string exeDirectory = AppContext.BaseDirectory;
            string launcherScriptPath = Path.Combine(exeDirectory, launcherScriptName);
            Console.WriteLine($"[Main] {launcherScriptName} path: {launcherScriptPath}");

            if (!File.Exists(launcherScriptPath))
            {
                Console.Error.WriteLine($"[Main] Error: {launcherScriptName} not found at '{launcherScriptPath}'.");
                Environment.Exit(1);
            }

            // Start the script with keep windows open
            try
            {
                LaunchServerLauncher(launcherScriptPath);
                Console.WriteLine("[Main] Waiting for the server to start...");
                bool serverStarted = false;
                for (int i = 0; i < 15; i++) // 
                {
                    if (IsPortInUse(targetPort))
                    {
                        Console.WriteLine($"[Main] Server started on port {targetPort} after {i+1} seconds.");
                        serverStarted = true;
                        Thread.Sleep(2000); // Wait for 2 seconds before proceeding for server to fully start
                        break;
                    }
                    else
                    {
                        Console.WriteLine($"[Main] Server not yet started. Checking again in 1 second...");
                        Thread.Sleep(1000);
                    }
                }
            
                if (!serverStarted)
                {
                    Console.Error.WriteLine("[Main] Error: Server did not start within 15 seconds.");
                    Environment.Exit(1);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Main] Error launching server: {ex.Message}");
                Environment.Exit(1);
            }
            
        }

        // 4. 与服务器通信
        try
        {
            var postFields = string.Join(" ", args);
            var content = new StringContent(postFields, Encoding.UTF8, "application/x-www-form-urlencoded");

            HttpResponseMessage response = await client.PostAsync("http://127.0.0.1:8572", content); 

            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine("Success: Straycat Resampled Sucessfully");
            } 
            else if (response.StatusCode == System.Net.HttpStatusCode.BadRequest)
            {
                Console.Error.WriteLine("Error: StrayCat got an incorrect amount of arguments or the arguments were out of order. Please check the input data before continuing.");
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.InternalServerError)
            {
                Console.Error.WriteLine($"Error: An Internal Error occured in StraycatServer. Check your voicebank wav files to ensure they are the correct format. More details:\n{await response.Content.ReadAsStringAsync()}");
            }
            else
            {
                Console.Error.WriteLine($"Error: Straycat returned {response.StatusCode}");
            }
        }
        catch (HttpRequestException ex)
        {
            Console.Error.WriteLine($"Request exception: {ex.Message}\nIs straycat_server running?");
        }
    }

    static bool IsPortInUse(int port)
    {
        // 定义检测端口
        Console.WriteLine($"[IsPortInUse] Checking if port {port} is in use...");
        try
        {
            using (var client = new TcpClient())
            {
                var result = client.BeginConnect("127.0.0.1", port, null, null);
                var success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(100));

                if (success)
                {
                    client.EndConnect(result);
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
        catch
        {
            Console.WriteLine($"[IsPortInUse] Port {port} is not in use or an error occurred while checking.");
            return false;
        }
    }

    // 定义服务器启动器启动器
    static void LaunchServerLauncher(string launcherScriptPath)
    {
        Console.WriteLine($"[LaunchServerLauncher] Attempting to launch server launcher.");

        // 初始化 command 和 arguments
        string command = "";
        string arguments = "";

        if (OperatingSystem.IsWindows())
        {
            // 使用 cmd /K 来启动一个新的 CMD 窗口并保持打开状态
            command = "cmd";
            arguments = $"/K python \"{launcherScriptPath}\"";
        }
        else if (OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
        {
            // 尝试几种常见的终端（可以根据需要添加更多）
            string[] terminals = { "gnome-terminal", "konsole", "xterm", "terminator", "xfce4-terminal" };
            bool terminalFound = false; // 添加一个标志位，用于记录是否找到可用的终端
            foreach (var terminal in terminals)
            {
                try
                {
                    // 检查终端是否存在
                    Process.Start(new ProcessStartInfo { FileName = terminal, Arguments = "--version", RedirectStandardOutput = true, UseShellExecute = false }).WaitForExit();
                   
                    command = terminal;

                    if (terminal == "gnome-terminal")
                    {
                        arguments = $"-- bash -c \"python '{launcherScriptPath}'; exec bash\"";
                    }
                    else
                    {
                        arguments = $"-e bash -c \"python '{launcherScriptPath}'; exec bash\"";
                    }
                    Console.WriteLine($"[LaunchServerLauncher] Using terminal: {command} {arguments}");
                    terminalFound = true; // 找到可用的终端，设置标志位
                    break; // 找到可用的终端，跳出循环
                }
                catch (Exception)
                {
                    // 如果找不到该终端，继续尝试下一个
                    continue;
                }
            }
            if (!terminalFound)
            {
                throw new Exception("No suitable terminal found.");
            }

        }
        else
        {
            throw new PlatformNotSupportedException("Unsupported operating system.");
        }


        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = command,
            Arguments = arguments,
            UseShellExecute = true, // 使用 shell 执行 (允许 cmd /K 或终端的 -e 选项)
            CreateNoWindow = false, // 创建新窗口
            WorkingDirectory = Path.GetDirectoryName(launcherScriptPath)
        };

        // 启动进程
        try
        {
            Process.Start(startInfo);
            Console.WriteLine($"[LaunchServerLauncher] Server launcher process started.");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[LaunchServerLauncher] Error starting server launcher: {ex}");
            throw; // 重新抛出异常
        }
    }
}
