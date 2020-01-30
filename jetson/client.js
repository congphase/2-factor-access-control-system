var pyshell = require('python-shell');
var fs = require('fs');
let _uid = '';
let _pwd = '';


if (process.argv[2] == null) {
    console.log("IP parameter can't empty");
    process.exit();
}
var socket = require('socket.io-client')('http://' + process.argv[2] + ':8000');

socket.on('clientTest', function (data) {
    console.log('data :', data);
});

socket.on('idToJetson', function (data) {
    console.log('data: ', data);
    
    let options = {
        mode: 'text',
        args: [data.id]
    };
    pyshell.PythonShell.run('../get_id_from_client_js.py', options, function (err, pythonResult) {
        if (err) throw err;
    });
    _uid = data.uid;
    _pwd = data.pwd;
});

function intervalFunc() {
    try {
        let content = fs.readFileSync('/home/gate/lffd-dir/msg_buffer.txt', 'utf8');
        if (content != '') {
            console.log(content);
            socket.emit('faceRecogn', {
                result: content,
                uid: _uid,
                pwd: _pwd
            });
            fs.writeFile('/home/gate/lffd-dir/msg_buffer.txt', '', function () {});
            _uid = '';
            _pwd = '';
        }
    } catch (e) {
        console.log('Error:', e.stack);
    }
}
setInterval(intervalFunc, 1000);
