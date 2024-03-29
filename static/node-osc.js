const OSC = require('osc-js')

const config = { udpClient: { port: 5282 } }
const osc = new OSC({ plugin: new OSC.BridgePlugin(config) })

osc.open() // start a WebSocket server on port 5282