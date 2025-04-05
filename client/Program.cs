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
        if (args.Length < 1)
        {
            Console.Error.WriteLine("[Main] No arguments provided.");
            Environment.Exit(1);
        }

        // Trying to dectect the port
        Console.WriteLine($"[Main] Checking if port {targetPort} is in use...");
        if (!IsPortInUse(targetPort))
        {
            Console.WriteLine($"[Main] Port {targetPort} is not in use. Attempting to launch server.");

        // Get the path for launching the launch_server.py script
        string exeDirectory = AppContext.BaseDirectory;
        string launcherScriptPath = Path.Combine(exeDirectory, launcherScriptName);

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
                for (int i = 0; i < 90; i++)
            {
                if (IsPortInUse(targetPort))
                {
                    Console.WriteLine($"[Main] Server started on port {targetPort} after {i + 1} seconds.");
                    serverStarted = true;
                    break;
                }
                else
                {
                    await Task.Delay(1000);
                }
            }

            if (!serverStarted)
            {
                    Console.Error.WriteLine("[Main] Error: Server did not start within 90 seconds.");
                Environment.Exit(1);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Main] Error launching server: {ex.Message}");
            Environment.Exit(1);
        }
    }

        // Communicate with server (From Straycat original code)
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

    // Define port dectecting method
    static bool IsPortInUse(int port)
    {
        try
        {
            using (var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                IAsyncResult result = socket.BeginConnect("127.0.0.1", port, null, null);
                bool success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(25)); // Try 25ms

                if (success)
                {
                    socket.EndConnect(result);
                    return true;
                }
                else
                {
                    socket.Close(); // Ensure the socket is closed if the connection fails
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

    // Define server launcher launching method
    static void LaunchServerLauncher(string launcherScriptPath)
    {
        Console.WriteLine($"[LaunchServerLauncher] Attempting to launch server launcher.");

        // Initialize command and arguments based on operating system
        string command = "";
        string arguments = "";

        if (OperatingSystem.IsWindows())
        {
            // Open a command prompt with /K option to keep windows open
            command = "cmd";
            arguments = $"/K python \"{launcherScriptPath}\"";
        }
        else if (OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
        {
            // Try some other mainstream terminals (Can add more base on need)
            string[] terminals = { "gnome-terminal", "konsole", "xterm", "terminator", "xfce4-terminal" };
            bool terminalFound = false; // Use this flag to track if a suitable terminal is found

            foreach (var terminal in terminals)
            {
                try
                {
                    // Check if the terminal is available
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
                    terminalFound = true; // Find a suitable terminal and flip the flag
                    break;
                }
                catch (Exception)
                {
                    // Try next option if can find one
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
            UseShellExecute = true, // Run with shell
            CreateNoWindow = false, // Create new window
            WorkingDirectory = Path.GetDirectoryName(launcherScriptPath)
        };

        // Start the process
        try
        {
            Process.Start(startInfo);
            Console.WriteLine($"[LaunchServerLauncher] Server launcher process started.");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[LaunchServerLauncher] Error starting server launcher: {ex}");
            throw;
        }
    }
}
