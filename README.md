# Running the Project with Docker

This section provides instructions to build and run the project using Docker.

## Prerequisites

- Ensure Docker and Docker Compose are installed on your system.
- The project uses Python 3.9 as specified in the Dockerfile.

## Build and Run Instructions

1. Build the Docker image and start the services using Docker Compose:

   ```bash
   docker-compose up --build
   ```

2. The application will be accessible at `http://localhost:8000`.

## Configuration

- The application exposes port `8000` as defined in the Docker Compose file.
- If required, you can define environment variables in a `.env` file and uncomment the `env_file` line in the `docker-compose.yml` file.

## Notes

- The `requirements.txt` file lists the Python dependencies for the project.
- The `module_service_proto.proto` file is included for protocol buffer definitions.

For further details, refer to the project documentation or contact the maintainers.