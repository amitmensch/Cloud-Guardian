/**
 * CloudGuardian Phase 4: Operational Dashboard Real-Time Telemetry Client
 * Handles WebSocket feeds, Canvas metric curves, animated risk dial, and chaos triggers.
 */

class CloudGuardianClient {
    constructor() {
        this.socket = null;
        this.historyLength = 40;
        this.metricHistory = {
            cpu: [],
            ram: [],
            latency: [],
            logs: []
        };
        this.initCanvas();
        this.initEventListeners();
        this.connectWebSocket();
    }

    initCanvas() {
        this.metricCanvas = document.getElementById('metricCanvas');
        this.logCanvas = document.getElementById('logCanvas');
        this.mCtx = this.metricCanvas ? this.metricCanvas.getContext('2d') : null;
        this.lCtx = this.logCanvas ? this.logCanvas.getContext('2d') : null;
        
        // Handle High-DPI displays
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        [this.metricCanvas, this.logCanvas].forEach(cvs => {
            if (!cvs) return;
            const rect = cvs.getBoundingClientRect();
            cvs.width = rect.width * window.devicePixelRatio;
            cvs.height = rect.height * window.devicePixelRatio;
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        
        try {
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                const badge = document.getElementById('connBadge');
                if (badge) {
                    badge.className = 'badge badge-emerald';
                    badge.innerHTML = '<span class="badge-pulse"></span> WS LIVE';
                }
            };

            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleTelemetry(data);
            };

            this.socket.onclose = () => {
                const badge = document.getElementById('connBadge');
                if (badge) {
                    badge.className = 'badge badge-amber';
                    badge.innerHTML = '<span class="badge-pulse"></span> RECONNECTING';
                }
                setTimeout(() => this.connectWebSocket(), 2000);
            };

            this.socket.onerror = () => {
                if (this.socket) this.socket.close();
            };
        } catch (e) {
            console.error('WebSocket connection error:', e);
        }
    }

    handleTelemetry(data) {
        // 1. Update Risk Dial
        const riskPct = data.risk_probability || 0;
        const riskEl = document.getElementById('riskValue');
        const gaugeCircle = document.getElementById('gaugeCircle');
        
        if (riskEl) riskEl.textContent = `${Math.round(riskPct)}%`;
        
        if (gaugeCircle) {
            const circumference = 2 * Math.PI * 85; // r=85
            const offset = circumference - (riskPct / 100) * circumference;
            gaugeCircle.style.strokeDashoffset = offset;

            // Change color dynamically
            if (riskPct < 30) {
                gaugeCircle.style.stroke = 'var(--accent-emerald)';
            } else if (riskPct < 70) {
                gaugeCircle.style.stroke = 'var(--accent-amber)';
            } else {
                gaugeCircle.style.stroke = 'var(--accent-rose)';
            }
        }

        // 2. Update Stats
        const ttfEl = document.getElementById('ttfValue');
        if (ttfEl) ttfEl.textContent = `${(data.time_to_failure_minutes || 0).toFixed(1)} mins`;

        const pValEl = document.getElementById('pValValue');
        if (pValEl) pValEl.textContent = (data.p_value || 0).toFixed(4);

        const mseEl = document.getElementById('mseValue');
        if (mseEl) mseEl.textContent = (data.reconstruction_error || 0).toFixed(5);

        const tickEl = document.getElementById('tickValue');
        if (tickEl) tickEl.textContent = `#${data.tick_index || 0}`;

        // 3. Update Chart Buffers
        const m = data.metrics || {};
        this.pushHistory(this.metricHistory.cpu, m.cpu_utilization_pct || 20);
        this.pushHistory(this.metricHistory.ram, m.memory_utilization_pct || 40);
        this.pushHistory(this.metricHistory.latency, (m.database_wait_latency_ms || 10) / 2); // scale for visibility
        this.pushHistory(this.metricHistory.logs, data.log_burst_rate || 5);

        this.renderCharts();

        // 4. Update Explainability Root Causes
        if (data.explainability && data.explainability.top_contributors) {
            this.renderExplainability(data.explainability);
        }

        // 5. Update Counterfactual Guidance
        if (data.counterfactual) {
            const cfText = document.getElementById('cfText');
            const runbookText = document.getElementById('runbookText');
            if (cfText) cfText.textContent = data.counterfactual.plain_english_counterfactual || 'Operating within normal envelope.';
            if (runbookText) runbookText.textContent = data.counterfactual.operational_runbook || 'No action required.';
        }

        // 6. Update 3-Gate Defense Status
        if (data.governance) {
            this.updateGateBadges(data.governance);
        }

        // 7. Update Alert Feed if alert dispatched
        if (data.alert) {
            this.addAlertItem(data.alert);
        }
    }

    pushHistory(arr, val) {
        arr.push(val);
        if (arr.length > this.historyLength) arr.shift();
    }

    renderCharts() {
        this.drawCurve(this.mCtx, this.metricCanvas, [
            { data: this.metricHistory.cpu, color: '#3B82F6', label: 'CPU' },
            { data: this.metricHistory.ram, color: '#8B5CF6', label: 'RAM' },
            { data: this.metricHistory.latency, color: '#F43F5E', label: 'DB Latency' }
        ], 100);

        this.drawCurve(this.lCtx, this.logCanvas, [
            { data: this.metricHistory.logs, color: '#06B6D4', label: 'Log Events / sec' }
        ], 60);
    }

    drawCurve(ctx, cvs, series, maxVal) {
        if (!ctx || !cvs) return;
        const w = cvs.width;
        const h = cvs.height;

        ctx.clearRect(0, 0, w, h);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let y = 0; y < h; y += h / 4) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        series.forEach(s => {
            if (s.data.length < 2) return;
            const step = w / (this.historyLength - 1);

            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();

            s.data.forEach((val, i) => {
                const norm = Math.min(val / maxVal, 1.0);
                const x = i * step;
                const y = h - norm * (h - 20) - 10;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            ctx.stroke();
        });
    }

    renderExplainability(expl) {
        const container = document.getElementById('rootCausesContainer');
        if (!container) return;
        container.innerHTML = '';

        expl.top_contributors.slice(0, 3).forEach(c => {
            const item = document.createElement('div');
            item.className = 'cause-item';
            item.innerHTML = `
                <div class="cause-header">
                    <span class="cause-name">${c.channel_name}</span>
                    <span class="cause-pct">${c.attribution_percentage}%</span>
                </div>
                <div class="cause-bar-bg">
                    <div class="cause-bar-fill" style="width: ${c.attribution_percentage}%"></div>
                </div>
            `;
            container.appendChild(item);
        });

        const diagText = document.getElementById('diagSummaryText');
        if (diagText) diagText.textContent = expl.plain_english_summary;
    }

    updateGateBadges(gov) {
        const g1 = document.getElementById('gate1Verdict');
        const g2 = document.getElementById('gate2Verdict');
        const g3 = document.getElementById('gate3Verdict');

        if (g1) {
            g1.textContent = gov.gate1_fdr_pass ? 'PASS' : 'BLOCKED';
            g1.style.color = gov.gate1_fdr_pass ? 'var(--accent-emerald)' : 'var(--accent-rose)';
        }
        if (g2) {
            g2.textContent = gov.gate2_hysteresis_pass ? 'PASS (3/5)' : 'WAIT (1/5)';
            g2.style.color = gov.gate2_hysteresis_pass ? 'var(--accent-emerald)' : 'var(--accent-amber)';
        }
        if (g3) {
            g3.textContent = gov.gate3_cooldown_pass ? 'READY' : 'MUTED';
            g3.style.color = gov.gate3_cooldown_pass ? 'var(--accent-emerald)' : 'var(--accent-purple)';
        }
    }

    addAlertItem(alert) {
        const feed = document.getElementById('alertFeed');
        if (!feed) return;

        const item = document.createElement('div');
        item.className = 'alert-item';
        item.innerHTML = `
            <div class="alert-time">${new Date().toLocaleTimeString()} • ${alert.severity} • Host: ${alert.host_id}</div>
            <div class="alert-msg">${alert.message}</div>
        `;
        feed.prepend(item);
    }

    initEventListeners() {
        document.querySelectorAll('.chaos-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const scenario = btn.getAttribute('data-scenario');
                this.triggerChaos(scenario, btn);
            });
        });
    }

    async triggerChaos(scenario, btnElement) {
        btnElement.classList.add('active');
        btnElement.style.opacity = '0.5';

        try {
            const resp = await fetch('/api/chaos/inject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenario })
            });
            const data = await resp.json();
            console.log('Injected chaos:', data);
        } catch (e) {
            console.error('Chaos injection request failed:', e);
        } finally {
            setTimeout(() => {
                btnElement.classList.remove('active');
                btnElement.style.opacity = '1';
            }, 1200);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.cloudGuardian = new CloudGuardianClient();
});
