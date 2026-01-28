from gradio_ui import interface

def main():
   # Launch the app with specified server name and port
    interface.launch(server_name="localhost", server_port=7860)

if __name__ == "__main__":
    main()
