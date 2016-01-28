/**
 * KIS Image Downloader
 *
 */

const secret = require("./secret.json");
const image_dir = "images";
const extension = ".png";
const retry = 3;
const concurrent = 4;

const uuid = process.argv[2] || null;
console.log("UUID:", uuid);

var deleted_uuid_set = new Set([
  "7a8f76ee-d221-41f0-8c51-7215ee39381c",
  "e8de54c8-cd0a-4170-aae1-26e65825313e",
  "1ce14625-c128-4798-9415-5e1b195a44af",
  "6ee79970-6cd9-461e-93a0-1c284aaa0575",
  "be147417-1a03-46f6-838f-68205dbdcd0f"
]);

var uuid_set = new Set();

var req = require("request-promise");
var fs = require("fs");
var path = require("path");

var fail_counter = 0;

req({
  uri: "https://kis.appcloud.info/api/list",
  qs: {
    secret: secret.key
  }
}).then(res => {
  var data = JSON.parse(res);
  console.log("size:", data.list.length);

  // テストで発生したゴミデータの除去
  var list = data.list.filter(item => {
    var prefix_id = item.split(":")[0];
    var res = false;
    if (uuid == null) {
      res = deleted_uuid_set.has(prefix_id) === false;
    } else {
      res = prefix_id === uuid;
    }
    if (res) {
      uuid_set.add(prefix_id);
    }
    return res;
  });
  console.log("available:", list.length);
  console.log("available ids:", uuid_set.size);

  var tasks = list.map(item => {
    var fail = retry;
    var download = (() =>
      req({
        uri: `https://kis.appcloud.info/api/${item}${extension}`,
        resolveWithFullResponse: true,
        encoding: null
      }).catch(err => {
        console.log("request error.");
        if (fail-- > 0) {
          return download();
        } else {
          throw err;
        }
      })
    );
    return download;
  });

  console.log("========================================");
  var index = 0;
  var counter = 0;

  var limit = Math.ceil(tasks.length / concurrent);
  var tasks_list = []
  for (var i = 0; i < concurrent; i++) {
    if (i < concurrent - 1) {
      tasks_list[i] = tasks.slice(limit * i, limit * (i + 1));
    } else {
      tasks_list[i] = tasks.slice(limit * i);
    }
  }

  var promises = tasks_list.map(task_list => {
    return task_list.reduce((a, b) => {
      var item = list[index++];
      var out = [];

      return a.then(b).then(response => {
        out.push(`${++counter}/${tasks.length}`);
        out.push("Status: " + response.statusCode);

        if (response.statusCode !== 200) {
          throw new Error("failed.");
        }

        var content_type = response.headers["content-type"];
        out.push("Content-Type:" + content_type);

        if (content_type !== "image/" + extension.slice(1)) {
          throw new Error("failed.");
        }

        out.push(`Size: ${response.body.length / 1024} KiB`);

        // save to image file
        return new Promise((resolve, reject) => {
          var file = item.split(":").join("_") + extension;
          out.push("filename: " + file);

          fs.writeFile(path.join(image_dir, file), response.body, {
            mode: 0o644
          }, err => {
            if (err) {
              reject(err);
              return;
            }
            resolve();
          });
        });
      }).then(() => {
        out.push("saved.\n----------------------------------------");
        console.log(out.join("\n"));
      }).catch(err => {
        fail_counter++;
        console.log(err);
      });
    }, Promise.resolve());
  });

  return Promise.all(promises);
}).then(() => {
  console.log("failed:", fail_counter);
  console.log("finished.");
});
