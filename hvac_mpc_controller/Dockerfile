FROM ghcr.io/edwardgarza/mpc-hvac-controller:latest

COPY run.sh /
COPY generate_config.py /

# Make run.sh executable
RUN chmod +x /run.sh

CMD ["/run.sh"] 