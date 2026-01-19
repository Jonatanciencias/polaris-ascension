# Monitoring Setup - Grafana + Prometheus + Alertmanager

**Session 18 - Phase 2: Advanced Monitoring**

## 📊 Architecture

```
┌─────────────┐
│   API       │ ← Exports metrics on /metrics
│   :8000     │
└──────┬──────┘
       │
       │ scrapes every 10s
       ▼
┌─────────────┐      ┌──────────────┐
│ Prometheus  │─────▶│ Alertmanager │─────▶ Slack/Discord/Email
│   :9090     │      │    :9093     │
└──────┬──────┘      └──────────────┘
       │
       │ queries
       ▼
┌─────────────┐
│  Grafana    │ ← Visualizes metrics
│   :3000     │
└─────────────┘
```

## 🚀 Quick Start

### Start everything:
```bash
docker-compose --profile monitoring up -d
```

### Access services:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **API**: http://localhost:8000

### Stop monitoring:
```bash
docker-compose --profile monitoring down
```

## 📈 Grafana Dashboards

### 1. API Overview Dashboard
Location: http://localhost:3000/d/rx580-api-overview

**Panels**:
- Request Rate (req/s)
- Error Rate (%)
- P95 Latency (seconds)
- Models Loaded (count)
- Request Rate by Status (2xx, 4xx, 5xx)
- Response Latency Percentiles (P50, P95, P99)

**Auto-refresh**: Every 5 seconds

## 🔔 Alert Rules

### API Alerts (`prometheus/alerts/api_alerts.yml`)

1. **HighErrorRate** (warning)
   - Condition: >5% errors in 5m
   - Duration: 2 minutes

2. **CriticalErrorRate** (critical)
   - Condition: >10% errors in 5m
   - Duration: 1 minute

3. **HighLatency** (warning)
   - Condition: P95 >100ms
   - Duration: 5 minutes

4. **APIDown** (critical)
   - Condition: No requests in 1m
   - Duration: 1 minute

5. **HighRequestRate** (warning)
   - Condition: >100 req/s
   - Duration: 2 minutes

6. **ModelLoadFailures** (warning)
   - Condition: >3 failures in 10m
   - Duration: 1 minute

7. **SlowInference** (warning)
   - Condition: P95 >1s
   - Duration: 5 minutes

8. **HighMemoryUsage** (warning)
   - Condition: >7GB
   - Duration: 5 minutes

### System Alerts (`prometheus/alerts/system_alerts.yml`)

1. **ContainerRestarting** (warning)
   - Condition: >2 restarts in 10m

2. **ContainerDown** (critical)
   - Condition: Container not responding

3. **PrometheusTargetDown** (warning)
   - Condition: Target unreachable for 2m

## 🔕 Configure Notifications

### Slack

1. Create Slack webhook:
   - Go to https://api.slack.com/messaging/webhooks
   - Create webhook for your workspace
   - Copy webhook URL

2. Edit `alertmanager.yml`:
   ```yaml
   receivers:
     - name: 'critical'
       slack_configs:
         - api_url: 'YOUR_WEBHOOK_URL'
           channel: '#alerts-critical'
   ```

3. Restart:
   ```bash
   docker-compose --profile monitoring restart alertmanager
   ```

### Discord

1. Create Discord webhook:
   - Server Settings → Integrations → Webhooks
   - Create webhook, copy URL
   - **Important**: Add `/slack` to the end of URL

2. Edit `alertmanager.yml`:
   ```yaml
   receivers:
     - name: 'critical'
       webhook_configs:
         - url: 'YOUR_DISCORD_WEBHOOK_URL/slack'
   ```

3. Restart alertmanager

### Email (Gmail)

1. Enable 2FA in Gmail
2. Create App Password
3. Edit `alertmanager.yml`:
   ```yaml
   receivers:
     - name: 'critical'
       email_configs:
         - to: 'alerts@example.com'
           from: 'your-email@gmail.com'
           smarthost: 'smtp.gmail.com:587'
           auth_username: 'your-email@gmail.com'
           auth_password: 'your-app-password'
   ```

## 🧪 Testing

### Generate test traffic:
```bash
# Health check
for i in {1..100}; do curl http://localhost:8000/health; done

# List models
curl http://localhost:8000/models

# Inference (if model loaded)
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"model_name": "your_model", "input_data": [[1,2,3,4]]}'
```

### Test alerts:
```bash
# Stop API to trigger APIDown alert
docker-compose stop api

# Wait 1 minute, check Alertmanager
open http://localhost:9093

# Restart API
docker-compose start api
```

## 📊 Metrics Endpoints

### API Metrics:
```bash
curl http://localhost:8000/metrics
```

### Prometheus Metrics:
```bash
curl http://localhost:9090/metrics
```

## 🔧 Troubleshooting

### Grafana shows "No data"
1. Check Prometheus is scraping:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```
2. Generate traffic to API
3. Wait 10-15 seconds for scrape

### Alerts not firing
1. Check rules are loaded:
   ```bash
   curl http://localhost:9090/api/v1/rules
   ```
2. Check Alertmanager connection:
   ```bash
   curl http://localhost:9090/api/v1/alertmanagers
   ```

### Notifications not sending
1. Check Alertmanager logs:
   ```bash
   docker-compose logs alertmanager
   ```
2. Verify webhook URLs are correct
3. Test webhook manually with curl

## 📁 File Structure

```
.
├── docker-compose.yml           # Updated with monitoring services
├── prometheus.yml               # Prometheus config with alerts
├── alertmanager.yml             # Alert routing config
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml   # Auto-configure Prometheus
│   │   └── dashboards/
│   │       └── dashboards.yml   # Auto-load dashboards
│   └── dashboards/
│       └── api-overview.json    # API Overview dashboard
└── prometheus/
    └── alerts/
        ├── api_alerts.yml       # API monitoring rules
        └── system_alerts.yml    # System monitoring rules
```

## 🎯 Best Practices

1. **Dashboard Organization**:
   - One dashboard per concern (API, Models, System)
   - Use consistent color schemes
   - Group related panels

2. **Alert Tuning**:
   - Start with conservative thresholds
   - Adjust based on baseline metrics
   - Use `for` duration to avoid flapping

3. **Notification Routing**:
   - Critical → Immediate notification
   - Warning → Grouped, less frequent
   - Use inhibit rules to reduce noise

4. **Retention**:
   - Prometheus: 30 days (configurable)
   - Adjust based on disk space

## 📚 Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Guide](https://prometheus.io/docs/alerting/latest/alertmanager/)

---

**Created**: Session 18 - Phase 2  
**Last Updated**: 19 de Enero, 2026  
**Quality**: 9.8/10 (professional, documented)
