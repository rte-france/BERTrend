# Job Scheduler - Docker Setup

A FastAPI-based job scheduler with persistent SQLite storage and ProcessPoolExecutor, containerized with Docker.

## 📁 Project Structure

```
job-scheduler/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore patterns
├── data/                  # SQLite database (persistent)
│   └── jobs.sqlite
└── logs/                  # Application logs (optional)
```

## 🚀 Quick Start

### 1. Build and Run

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Access the API

- **API Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

### 3. Create Your First Job

```bash
# Create a job that runs every minute
curl -X POST "http://localhost:8000/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test_job",
    "job_type": "interval",
    "function_name": "sample_job",
    "minutes": 1,
    "kwargs": {"message": "Hello from Docker!"}
  }'

# List all jobs
curl http://localhost:8000/jobs

# Check job details
curl http://localhost:8000/jobs/test_job
```

## 🛠️ Docker Commands

### Basic Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f scheduler

# Execute commands in container
docker-compose exec scheduler bash
```

### Database Management

```bash
# Backup database
docker-compose exec scheduler cp /app/data/jobs.sqlite /app/data/jobs_backup.sqlite

# Access database
docker-compose exec scheduler sqlite3 /app/data/jobs.sqlite
```

### Rebuilding

```bash
# Rebuild after code changes
docker-compose up -d --build

# Force rebuild
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoring

### Health Check

The container includes automatic health checks:

```bash
# Check container health
docker-compose ps

# Manual health check
curl http://localhost:8000/
```

### Logs

```bash
# View real-time logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# View logs from specific service
docker-compose logs scheduler
```

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - TZ=Europe/Paris          # Timezone
  - LOG_LEVEL=INFO           # Logging level
  - MAX_WORKERS=5            # ProcessPoolExecutor workers
  - DB_PATH=/app/data/jobs.sqlite
```

### Resource Limits

Adjust CPU and memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 1G
```

### Port Mapping

Change the exposed port:

```yaml
ports:
  - "8080:8000"  # Host:Container
```

## 💾 Data Persistence

The SQLite database is stored in `./data/jobs.sqlite` on your host machine, ensuring jobs persist across container restarts.

### Backup Strategy

```bash
# Manual backup
cp data/jobs.sqlite data/jobs_backup_$(date +%Y%m%d).sqlite

# Automated backup script
#!/bin/bash
docker-compose exec scheduler sqlite3 /app/data/jobs.sqlite ".backup '/app/data/backup_$(date +%Y%m%d_%H%M%S).sqlite'"
```

## 🔍 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

# Rebuild
docker-compose down
docker-compose up -d --build
```

### Database locked

```bash
# Stop all instances
docker-compose down

# Remove stale locks
rm data/jobs.sqlite-journal

# Restart
docker-compose up -d
```

### Permission issues

```bash
# Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

## 🚢 Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml scheduler

# Check services
docker stack services scheduler
```

### Using Kubernetes

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: job-scheduler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: scheduler
  template:
    metadata:
      labels:
        app: scheduler
    spec:
      containers:
      - name: scheduler
        image: job-scheduler:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: scheduler-data
```

## 📝 API Examples

### Cron Job (Daily at 9 AM Paris time)

```bash
curl -X POST "http://localhost:8000/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "daily_report",
    "job_type": "cron",
    "function_name": "report_generator",
    "cron_expression": "0 9 * * *",
    "kwargs": {"report_type": "daily"}
  }'
```

### Validate Cron Expression

```bash
curl -X POST "http://localhost:8000/cron/validate" \
  -H "Content-Type: application/json" \
  -d '{"expression": "*/15 * * * *"}'
```

### Get Cron Examples

```bash
curl http://localhost:8000/cron/examples
```

## 🔐 Security Considerations

For production:

1. **Add authentication**: Implement API key or OAuth
2. **Use HTTPS**: Add reverse proxy (nginx/traefik)
3. **Limit exposure**: Use internal networks
4. **Regular backups**: Automated database backups
5. **Update dependencies**: Keep packages current

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)

## 📄 License

This project is provided as-is for educational and commercial use.